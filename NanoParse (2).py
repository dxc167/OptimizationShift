# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np 
import matplotlib.pyplot as plt
from numpy.lib.scimath import sqrt
import tidy3d as td 
from tidy3d import web
import os
from scipy.optimize import curve_fit
from scipy import optimize
from scipy.interpolate import CubicSpline
import csv
import pandas as pd
import trimesh
from IPython.display import Image, display

def TMM_setup(n1, n2, n3, incident_angle, wavelength, tg, ts, ts_total, layering):
    # Calculates reflectivity and phase by TMM methods
    # n1 = prism refractive index
    # n2 = gold refractive index
    # n3 = sample refractive index
    # incident_angle = incident angle from substance 1 onto prism (From normal)
    # tg = gold thickness
    # ts = sample thickness per layer
    # ts_total = total sample thickness
    # layering = 'on' or 'off' to use multiple layers for thick samples

    Ep = n1**2
    a = (n1*np.sin(incident_angle))**2

    Qpk_p =  sqrt(Ep - a) / Ep
    Qpk_s =  Qpk_p * Ep      

    # compute gold TMM
    n_gold = n2
    Eg = n_gold**2

    Bgk = ((2 * np.pi * tg) / wavelength) * sqrt(Eg - a)

    Qgk_p = sqrt(Eg - a) / Eg
    Qgk_s = Qgk_p * Eg

    gold_M11 = np.cos(Bgk)
    gold_M12p = (-1j * np.sin(Bgk)) / Qgk_p
    gold_M21p = -1j * Qgk_p * np.sin(Bgk)
    gold_M22 = np.cos(Bgk)

    gold_M12s = (-1j * np.sin(Bgk)) / Qgk_s
    gold_M21s = -1j * Qgk_s * np.sin(Bgk)

    gold_matrix_p = np.array([[gold_M11, gold_M12p],
                              [gold_M21p, gold_M22]])

    gold_matrix_s = np.array([[gold_M11, gold_M12s],
                              [gold_M21s, gold_M22]])

    # calculate transfer matrix for sample
    n_sample = n3
    Es = n_sample**2

    Bsk = ((2 * np.pi * ts) / wavelength) * sqrt(Es - a)

    Qsk_p = sqrt(Es - a) / Es
    Qsk_s = sqrt(Es - a)

    sample_M11 = np.cos(Bsk)
    sample_M12p = (-1j * np.sin(Bsk)) / Qsk_p
    sample_M21p = -1j * Qsk_p * np.sin(Bsk)
    sample_M22 = np.cos(Bsk)

    sample_M12s = (-1j * np.sin(Bsk)) / Qsk_s
    sample_M21s = -1j * Qsk_s * np.sin(Bsk)

    sample_matrix_p = np.array([[sample_M11, sample_M12p],
                                [sample_M21p, sample_M22]])

    sample_matrix_s = np.array([[sample_M11, sample_M12s],
                                [sample_M21s, sample_M22]])

    if layering == 'off':
        # Multiply matrices
        TMM_matrix_p = np.dot(sample_matrix_p, gold_matrix_p)
        Mp = TMM_matrix_p
        M11p = Mp[0,0]
        M12p = Mp[0,1]
        M21p = Mp[1,0]
        M22p = Mp[1,1]

        TMM_matrix_s = np.dot(sample_matrix_s, gold_matrix_s)
        Ms = TMM_matrix_s
        M11s = Ms[0,0]
        M12s = Ms[0,1]
        M21s = Ms[1,0]
        M22s = Ms[1,1]

    else:
        # Multiply matrices with layering
        N_samples = int(ts_total / ts)
        pmp = np.linalg.matrix_power(sample_matrix_p, N_samples)
        smp = np.linalg.matrix_power(sample_matrix_s, N_samples)
        TMM_matrix_p = np.dot(pmp, gold_matrix_p)
        TMM_matrix_s = np.dot(smp, gold_matrix_s)

        Ms = TMM_matrix_s
        M11s = Ms[0,0]
        M12s = Ms[0,1]
        M21s = Ms[1,0]
        M22s = Ms[1,1]    

        Mp = TMM_matrix_p
        M11p = Mp[0,0]
        M12p = Mp[0,1]
        M21p = Mp[1,0]
        M22p = Mp[1,1]

    # Calculate fresnel coefficients
    qN_p = Qsk_p
    q1_p = Qpk_p
    r_p = ((M11p + M12p * qN_p) * q1_p - (M21p + M22p * qN_p)) / \
          ((M11p + M12p * qN_p) * q1_p + (M21p + M22p * qN_p))
    p_phase = np.angle(r_p)
    p_reflectivity = (np.abs(r_p))**2

    qN_s = Qsk_s
    q1_s = Qpk_s
    r_s = ((M11s + M12s * qN_s) * q1_s - (M21s + M22s * qN_s)) / \
          ((M11s + M12s * qN_s) * q1_s + (M21s + M22s * qN_s))
    s_phase = np.angle(r_s)
    s_reflectivity = (np.abs(r_s))**2

    return s_phase, p_phase, s_reflectivity, p_reflectivity

def scan_angle(start_angle, end_angle, scans, n1, n2, n3, wavelength, tg, ts, ts_total, layering, plot):
    angles = np.linspace(start_angle, end_angle, scans)

    sphas = []
    pphas = []
    sref = []
    pref = []

    for angle in angles:
        # Convert degrees to radians
        incident_angle = np.radians(angle)

        spha, ppha, sre, pre = TMM_setup(n1, n2, n3, incident_angle, wavelength, tg, ts, ts_total, layering)
        sphas.append(spha)
        pphas.append(ppha)
        sref.append(sre)
        pref.append(pre)

    sphas = np.array(sphas)
    pphas = np.array(pphas)
    sref = np.array(sref)
    pref = np.array(pref)

    # Unwrap the phase
    unw_pphas = np.degrees(np.unwrap(pphas))
    unw_sphas = np.degrees(np.unwrap(sphas))

    # Calculate Goos-Hänchen shifts
    gh_shift_p = ((-1 * wavelength) / (2 * np.pi)) * np.gradient(unw_pphas, angles)
    gh_shift_s = ((-1 * wavelength) / (2 * np.pi)) * np.gradient(unw_sphas, angles)
    diff_gh = gh_shift_p - gh_shift_s

    if plot == 'on':
        fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 12))  # Create a 3x3 grid of subplots

        # Wrapped S phase
        axes[0, 0].plot(angles, sphas)
        axes[0, 0].set_title("Wrapped S phase")

        # Unwrapped S phase
        axes[0, 1].plot(angles, unw_sphas)
        axes[0, 1].set_title("Unwrapped S phase")

        # Lateral Goos-Hänchen shift of S polarization
        axes[0, 2].plot(angles, gh_shift_s)
        axes[0, 2].set_title("Lateral Goos-Hänchen shift (S polarization)")

        # Wrapped P phase
        axes[1, 0].plot(angles, pphas)
        axes[1, 0].set_title("Wrapped P phase")

        # Unwrapped P phase
        axes[1, 1].plot(angles, unw_pphas)
        axes[1, 1].set_title("Unwrapped P phase")

        # Lateral Goos-Hänchen shift of P polarization
        axes[1, 2].plot(angles, gh_shift_p)
        axes[1, 2].set_title("Lateral Goos-Hänchen shift (P polarization)")

        # Differential Goos-Hänchen shift
        axes[2, 0].plot(angles, diff_gh)
        axes[2, 0].set_title("Differential Goos-Hänchen shift")

        # P reflectivity
        axes[2, 1].plot(angles, pref, label='P reflectivity')
        axes[2, 1].set_title("Reflectivity")
        axes[2, 1].plot(angles, sref, label='S reflectivity')
        axes[2, 1].legend()

        # S reflectivity
        axes[2, 2].plot(angles, sref)
        axes[2, 2].set_title("S reflectivity")

        # Adjust layout to prevent overlap
        plt.tight_layout()
        plt.show()
        print(np.min(diff_gh))

    return unw_pphas, unw_sphas, sphas, pphas, sref, pref, gh_shift_p, gh_shift_s, diff_gh, angles

def sample_scan(start_angle, end_angle, scans, n1, n2, start_index, end_index, steps, 
                wavelength, tg, ts, ts_total, layering, graphs):

    # Initialize lists to store outputs from the scan_angle function
    sphas_array = []
    pphas_array = []
    unw_pphas_array = []
    unw_sphas_array = []
    sref_array = []
    pref_array = []
    gh_shift_p_array = []
    gh_shift_s_array = []
    diff_gh_array = []
    labels = []

    # Initialize variable to store angles (assuming angles are the same for all n2)
    angles = None
    indices = np.linspace(start_index, end_index, steps)

    max_diff_gh_values = []  # To store maximum differential GH shift for each sample RI

    for i in range(len(indices)):
        n3 = indices[i]

        # Call the 'scan_angle' function with all the required parameters
        # Set 'plot' parameter to 'off' to suppress plotting within 'scan_angle'
        unw_pphas, unw_sphas, sphas, pphas, sref, pref, gh_shift_p, gh_shift_s, diff_gh, angles = scan_angle(
            start_angle, end_angle, scans, n1, n2, n3, wavelength, tg, ts, ts_total, layering, plot='off')

        # Append all outputs to their respective lists
        unw_pphas_array.append(unw_pphas)
        unw_sphas_array.append(unw_sphas)
        sphas_array.append(sphas)
        pphas_array.append(pphas)
        sref_array.append(sref)
        pref_array.append(pref)
        gh_shift_p_array.append(gh_shift_p)
        gh_shift_s_array.append(gh_shift_s)
        diff_gh_array.append(diff_gh)
        labels.append(f'n = {n3:.3f}')

        # Compute maximum absolute differential GH shift for this n3
        max_diff_gh = np.max(np.abs(diff_gh))
        max_diff_gh_values.append(max_diff_gh)

    # Plotting
    if graphs == 'complex':
        # After the loop, plot the desired values for each refractive index
        num_plots = 9  # Total number of plots
        num_cols = 3
        num_rows = int(np.ceil(num_plots / num_cols))

        # Create subplots
        fig, axs = plt.subplots(num_rows, num_cols, figsize=(18, num_rows * 4))
        axs = axs.flatten()  # Flatten the array of axes for easy indexing

        # Plot configurations (including unwrapped phases)
        plot_data = [
            (sphas_array, 'Wrapped S Phase vs Angles', 'Wrapped S Phase'),
            (pphas_array, 'Wrapped P Phase vs Angles', 'Wrapped P Phase'),
            (unw_sphas_array, 'Unwrapped S Phase vs Angles', 'Unwrapped S Phase'),
            (unw_pphas_array, 'Unwrapped P Phase vs Angles', 'Unwrapped P Phase'),
            (gh_shift_p_array, 'GH Shift P vs Angles', 'GH Shift P'),
            (gh_shift_s_array, 'GH Shift S vs Angles', 'GH Shift S'),
            (diff_gh_array, 'Differential GH Shift vs Angles', 'Differential GH Shift'),
            (sref_array, 'S Reflectivity vs Angles', 'S Reflectivity'),
            (pref_array, 'P Reflectivity vs Angles', 'P Reflectivity')
        ]

        # Loop over each plot configuration
        for ax_idx, (data_array, title, ylabel) in enumerate(plot_data):
            for data, label in zip(data_array, labels):
                axs[ax_idx].plot(angles, data, label=label)
            axs[ax_idx].set_title(title)
            axs[ax_idx].set_xlabel('Angle (degrees)')
            axs[ax_idx].set_ylabel(ylabel)
            axs[ax_idx].grid(True)
            # Place legend outside the plot area
            axs[ax_idx].legend(loc='upper left', bbox_to_anchor=(1, 1))

        # Hide any unused subplots (if any)
        for idx in range(len(plot_data), len(axs)):
            fig.delaxes(axs[idx])

        plt.tight_layout()
        plt.show()

    elif graphs == 'simple':
        # Only plot GH shift and P reflectivity graphs
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        # Plot GH shift P
        for data, label in zip(gh_shift_p_array, labels):
            axs[0].plot(angles, data, label=label)
        axs[0].set_title('GH Shift P vs Angles')
        axs[0].set_xlabel('Angle (degrees)')
        axs[0].set_ylabel('GH Shift P')
        axs[0].grid(True)
        axs[0].legend(loc='upper left', bbox_to_anchor=(1, 1))

        # Plot P Reflectivity
        for data, label in zip(pref_array, labels):
            axs[1].plot(angles, data, label=label)
        axs[1].set_title('P Reflectivity vs Angles')
        axs[1].set_xlabel('Angle (degrees)')
        axs[1].set_ylabel('P Reflectivity')
        axs[1].grid(True)
        axs[1].legend(loc='upper left', bbox_to_anchor=(1, 1))

        plt.tight_layout()
        plt.show()

    elif graphs == 'shift':
        # Plot maximum differential GH shift vs sample refractive index
        plt.figure(figsize=(8, 6))
        plt.plot(indices, max_diff_gh_values, marker='o')
        plt.title('Maximum Differential GH Shift vs Sample Refractive Index')
        plt.xlabel('Sample Refractive Index')
        plt.ylabel('Maximum Differential GH Shift')
        plt.grid(True)
        # Place legend outside the plot area (if needed)
        plt.legend(['Max Diff GH Shift'], loc='upper left', bbox_to_anchor=(1, 1))
        plt.tight_layout()


    elif graphs=='sen':
        

        
        plt.figure(figsize=(8, 6))
        plt.plot(indices, max_diff_gh_values, marker='o')
        plt.title('Maximum Differential GH Shift vs Sample Refractive Index')
        plt.xlabel('Sample Refractive Index')
        plt.ylabel('Maximum Differential GH Shift')
        plt.grid(True)
        # Place legend outside the plot area (if needed)
        plt.legend(['Max Diff GH Shift'], loc='upper left', bbox_to_anchor=(1, 1))
        plt.tight_layout()

        
        plt.show()

    # Compute the resonance angle from the pref_array
    res_ind = np.argmin(pref_array[0])
    res_ang = angles[res_ind]

    ang_int_sens = []
    int_int_sens = []
    gh_sens = []
    gh_peaks = []

    for i in range(len(indices)):
        gh_peak_ind = np.argmax(np.abs(diff_gh_array[i]))
        gh_peak = diff_gh_array[i][gh_peak_ind]

        # Shift position interrogation
        gh = diff_gh_array[i][res_ind]
        gh_sens.append(gh)

        # Shift peak position
        gh_peaks.append(gh_peak)

        # Intensity interrogation
        int_int = pref_array[i][res_ind]
        int_int_sens.append(int_int)

        ang_int = angles[np.argmin(pref_array[i])]
        ang_int_sens.append(ang_int)

    # Return the maximum absolute Goos-Hänchen shift (biggest positive or negative)
    max_gh_shift = np.max(np.abs(gh_peaks))

    return max_gh_shift

def scan_angle_index(start_angle, end_angle, scans, n1, start_index, end_index, steps, n3, wavelength, tg, ts, ts_total, layering, graphs):
    indices = np.linspace(start_index, end_index, steps)

    # Initialize lists to store outputs from the scan_angle function
    sphas_array = []
    pphas_array = []
    unw_pphas_array = []
    unw_sphas_array = []
    sref_array = []
    pref_array = []
    gh_shift_p_array = []
    gh_shift_s_array = []
    diff_gh_array = []
    labels = []

    max_diff_gh_values = []  # To store maximum differential GH shift for each film RI

    angles = None

    for i in range(len(indices)):
        n2 = indices[i]

        # Call the 'scan_angle' function
        unw_pphas, unw_sphas, sphas, pphas, sref, pref, gh_shift_p, gh_shift_s, diff_gh, angles = scan_angle(
            start_angle, end_angle, scans, n1, n2, n3, wavelength, tg, ts, ts_total, layering, plot='off')

        # Append all outputs to their respective lists
        unw_pphas_array.append(unw_pphas)
        unw_sphas_array.append(unw_sphas)
        sphas_array.append(sphas)
        pphas_array.append(pphas)
        sref_array.append(sref)
        pref_array.append(pref)
        gh_shift_p_array.append(gh_shift_p)
        gh_shift_s_array.append(gh_shift_s)
        diff_gh_array.append(diff_gh)
        labels.append(f'n = {n2:.3f}')

        # Compute maximum absolute differential GH shift for this n2
        max_diff_gh = np.max(np.abs(diff_gh))
        max_diff_gh_values.append(max_diff_gh)

    # Plotting
    if graphs == 'complex':
        # After the loop, plot the desired values for each refractive index
        num_plots = 9  # Total number of plots
        num_cols = 3
        num_rows = int(np.ceil(num_plots / num_cols))

        # Create subplots
        fig, axs = plt.subplots(num_rows, num_cols, figsize=(18, num_rows * 4))
        axs = axs.flatten()  # Flatten the array of axes for easy indexing

        # Plot configurations (including unwrapped phases)
        plot_data = [
            (sphas_array, 'Wrapped S Phase vs Angles', 'Wrapped S Phase'),
            (pphas_array, 'Wrapped P Phase vs Angles', 'Wrapped P Phase'),
            (unw_sphas_array, 'Unwrapped S Phase vs Angles', 'Unwrapped S Phase'),
            (unw_pphas_array, 'Unwrapped P Phase vs Angles', 'Unwrapped P Phase'),
            (gh_shift_p_array, 'GH Shift P vs Angles', 'GH Shift P'),
            (gh_shift_s_array, 'GH Shift S vs Angles', 'GH Shift S'),
            (diff_gh_array, 'Differential GH Shift vs Angles', 'Differential GH Shift'),
            (sref_array, 'S Reflectivity vs Angles', 'S Reflectivity'),
            (pref_array, 'P Reflectivity vs Angles', 'P Reflectivity')
        ]

        # Loop over each plot configuration
        for ax_idx, (data_array, title, ylabel) in enumerate(plot_data):
            for data, label in zip(data_array, labels):
                axs[ax_idx].plot(angles, data, label=label)
            axs[ax_idx].set_title(title)
            axs[ax_idx].set_xlabel('Angle (degrees)')
            axs[ax_idx].set_ylabel(ylabel)
            axs[ax_idx].grid(True)
            # Place legend outside the plot area
            axs[ax_idx].legend(loc='upper left', bbox_to_anchor=(1, 1))

        # Hide any unused subplots (if any)
        for idx in range(len(plot_data), len(axs)):
            fig.delaxes(axs[idx])

        plt.tight_layout()
        plt.show()

    elif graphs == 'simple':
        # Only plot GH shift and P reflectivity graphs
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        # Plot GH shift P
        for data, label in zip(gh_shift_p_array, labels):
            axs[0].plot(angles, data, label=label)
        axs[0].set_title('GH Shift P vs Angles')
        axs[0].set_xlabel('Angle (degrees)')
        axs[0].set_ylabel('GH Shift P')
        axs[0].grid(True)
        axs[0].legend(loc='upper left', bbox_to_anchor=(1, 1))

        # Plot P Reflectivity
        for data, label in zip(pref_array, labels):
            axs[1].plot(angles, data, label=label)
        axs[1].set_title('P Reflectivity vs Angles')
        axs[1].set_xlabel('Angle (degrees)')
        axs[1].set_ylabel('P Reflectivity')
        axs[1].grid(True)
        axs[1].legend(loc='upper left', bbox_to_anchor=(1, 1))

        plt.tight_layout()
        plt.show()

    elif graphs == 'shift':
        # Plot maximum differential GH shift vs film refractive index
        plt.figure(figsize=(8, 6))
        plt.plot(indices, max_diff_gh_values, marker='o')
        plt.title('Maximum Differential GH Shift vs Film Refractive Index')
        plt.xlabel('Film Refractive Index')
        plt.ylabel('Maximum Differential GH Shift')
        plt.grid(True)
        plt.legend(['Max Diff GH Shift'], loc='upper left', bbox_to_anchor=(1, 1))
        plt.tight_layout()
        plt.show()

    # Compute Goos-Hänchen shift at SPR and maximum absolute GH shift
    gh_sens = []
    gh_peaks = []

    for i in range(len(indices)):
        res_ind = np.argmin(pref_array[i])
        gh = diff_gh_array[i][res_ind]
        gh_sens.append(gh)

        # Maximum absolute GH shift
        gh_peak_ind = np.argmax(np.abs(diff_gh_array[i]))
        gh_peak = diff_gh_array[i][gh_peak_ind]
        gh_peaks.append(gh_peak)

    # Return the maximum absolute Goos-Hänchen shift (biggest positive or negative)
    max_gh_shift = np.max(np.abs(gh_peaks))

    return max_gh_shift

def scan_angle_mthick(start_angle, end_angle, scans, n1, n2, n3, start_thickness, end_thickness, steps, wavelength, ts, ts_total, layering, graphs):
    thicknesses = np.linspace(start_thickness, end_thickness, steps)

    # Initialize lists to store outputs from the scan_angle function
    sphas_array = []
    pphas_array = []
    unw_pphas_array = []
    unw_sphas_array = []
    sref_array = []
    pref_array = []
    gh_shift_p_array = []
    gh_shift_s_array = []
    diff_gh_array = []
    labels = []

    max_diff_gh_values = []  # To store maximum differential GH shift for each thickness

    angles = None

    for i in range(len(thicknesses)):
        tg = thicknesses[i]

        # Call the 'scan_angle' function
        unw_pphas, unw_sphas, sphas, pphas, sref, pref, gh_shift_p, gh_shift_s, diff_gh, angles = scan_angle(
            start_angle, end_angle, scans, n1, n2, n3, wavelength, tg, ts, ts_total, layering, plot='off')

        # Append all outputs to their respective lists
        unw_pphas_array.append(unw_pphas)
        unw_sphas_array.append(unw_sphas)
        sphas_array.append(sphas)
        pphas_array.append(pphas)
        sref_array.append(sref)
        pref_array.append(pref)
        gh_shift_p_array.append(gh_shift_p)
        gh_shift_s_array.append(gh_shift_s)
        diff_gh_array.append(diff_gh)
        labels.append(f'tg = {tg:.3e}')

        # Compute maximum absolute differential GH shift for this thickness
        max_diff_gh = np.max(np.abs(diff_gh))
        max_diff_gh_values.append(max_diff_gh)

    # Plotting
    if graphs == 'complex':
        # After the loop, plot the desired values for each thickness
        num_plots = 9  # Total number of plots
        num_cols = 3
        num_rows = int(np.ceil(num_plots / num_cols))

        # Create subplots
        fig, axs = plt.subplots(num_rows, num_cols, figsize=(18, num_rows * 4))
        axs = axs.flatten()  # Flatten the array of axes for easy indexing

        # Plot configurations (including unwrapped phases)
        plot_data = [
            (sphas_array, 'Wrapped S Phase vs Angles', 'Wrapped S Phase'),
            (pphas_array, 'Wrapped P Phase vs Angles', 'Wrapped P Phase'),
            (unw_sphas_array, 'Unwrapped S Phase vs Angles', 'Unwrapped S Phase'),
            (unw_pphas_array, 'Unwrapped P Phase vs Angles', 'Unwrapped P Phase'),
            (gh_shift_p_array, 'GH Shift P vs Angles', 'GH Shift P'),
            (gh_shift_s_array, 'GH Shift S vs Angles', 'GH Shift S'),
            (diff_gh_array, 'Differential GH Shift vs Angles', 'Differential GH Shift'),
            (sref_array, 'S Reflectivity vs Angles', 'S Reflectivity'),
            (pref_array, 'P Reflectivity vs Angles', 'P Reflectivity')
        ]

        # Loop over each plot configuration
        for ax_idx, (data_array, title, ylabel) in enumerate(plot_data):
            for data, label in zip(data_array, labels):
                axs[ax_idx].plot(angles, data, label=label)
            axs[ax_idx].set_title(title)
            axs[ax_idx].set_xlabel('Angle (degrees)')
            axs[ax_idx].set_ylabel(ylabel)
            axs[ax_idx].grid(True)
            # Place legend outside the plot area
            axs[ax_idx].legend(loc='upper left', bbox_to_anchor=(1, 1))

        # Hide any unused subplots (if any)
        for idx in range(len(plot_data), len(axs)):
            fig.delaxes(axs[idx])

        plt.tight_layout()
        plt.show()

    elif graphs == 'simple':
        # Only plot GH shift and P reflectivity graphs
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        # Plot GH shift P
        for data, label in zip(gh_shift_p_array, labels):
            axs[0].plot(angles, data, label=label)
        axs[0].set_title('GH Shift P vs Angles')
        axs[0].set_xlabel('Angle (degrees)')
        axs[0].set_ylabel('GH Shift P')
        axs[0].grid(True)
        axs[0].legend(loc='upper left', bbox_to_anchor=(1, 1))

        # Plot P Reflectivity
        for data, label in zip(pref_array, labels):
            axs[1].plot(angles, data, label=label)
        axs[1].set_title('P Reflectivity vs Angles')
        axs[1].set_xlabel('Angle (degrees)')
        axs[1].set_ylabel('P Reflectivity')
        axs[1].grid(True)
        axs[1].legend(loc='upper left', bbox_to_anchor=(1, 1))

        plt.tight_layout()
        plt.show()

    elif graphs == 'shift':
        # Plot maximum differential GH shift vs film thickness
        plt.figure(figsize=(8, 6))
        plt.plot(thicknesses, max_diff_gh_values, marker='o')
        plt.title('Maximum Differential GH Shift vs Film Thickness')
        plt.xlabel('Film Thickness (m)')
        plt.ylabel('Maximum Differential GH Shift')
        plt.grid(True)
        plt.legend(['Max Diff GH Shift'], loc='upper left', bbox_to_anchor=(1, 1))
        plt.tight_layout()
        plt.show()

    # Compute Goos-Hänchen shift at SPR and maximum absolute GH shift
    gh_sens = []
    gh_peaks = []

    for i in range(len(thicknesses)):
        res_ind = np.argmin(pref_array[i])
        gh = diff_gh_array[i][res_ind]
        gh_sens.append(gh)

        # Maximum absolute GH shift
        gh_peak_ind = np.argmax(np.abs(diff_gh_array[i]))
        gh_peak = diff_gh_array[i][gh_peak_ind]
        gh_peaks.append(gh_peak)

    # Return the maximum absolute Goos-Hänchen shift (biggest positive or negative)
    max_gh_shift = np.max(np.abs(gh_peaks))

    return max_gh_shift

def sp_calc(n_prism, n_metal, n_sample):
    mag_metal = np.abs(n_metal)**2
    mag_prism = np.abs(n_prism)**2
    mag_sample = np.abs(n_sample)**2

    mag_multi = mag_metal * mag_sample
    mag_add = mag_metal + mag_sample
    multi_add_rat = mag_multi / mag_add
    x1 = np.sqrt(multi_add_rat)

    resonance_angle1 = np.arcsin((1 / n_prism) * x1)

    rangle = np.degrees(resonance_angle1)

    print("The surface plasmon angle is", rangle)
    return rangle

def crit_thick(wavelength, n_metal):
    # From https://pubs.aip.org/apl/article/85/3/372/320272/Large-positive-and-negative-lateral-optical-beam
    real_metal = np.real(n_metal)
    imag_metal = np.imag(n_metal)
    A = wavelength / (4 * np.pi * imag_metal)
    Dcr = A * np.log((2 * imag_metal) / real_metal)
    return Dcr

def set_nano_sim(
    wavelength_range,
    Nfreqs,
    source, #choose PlaneWave or Gaussian
    height,#depth of each hole
    spacing,#spacing between holes
    diameter,#hole diamater
    t_base,#metal film thickness
    metal_material,#using JC gold for now
    background_material, #hole material
    prism_material,
    sample_material,
    #angle,
    pol_angle,
    run_t,
    grid_res,
    mesh_res,
    n_prism,
    hole,
    sim_height):
    
    
    #SET PARAMATERS 
    
    #n_prism = np.sqrt(prism_material)
    stack_point =0
    sp = stack_point
    
    
    min_wavelength, max_wavelength = wavelength_range
    #frequency and wavelength paramaters
    central_wavelength =np.mean(wavelength_range)
    lda0 = central_wavelength  # central wavelength
    freq0 = td.C_0 / lda0  # central frequency
    ldas = np.linspace(min_wavelength, max_wavelength, Nfreqs)  # wavelength range
    freqs = td.C_0 / ldas  # frequency range
    fwidth = 0.5 * (np.max(freqs) - np.min(freqs))  # width of the source frequency range

    #-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #define geometry for nanohole array
    h = height #depth of each hole 
    a = spacing #spacing between hole 
    d = diameter #diamater of hole
    t_base = t_base # thickness of base film

    material = metal_material
    hole_center = sp +h/2
    #CREATE UP TO FIVE HOLES FOR A UNIT CELL
    
    hole_1 = td.Structure(
        geometry=td.Cylinder(center=(0, 0, hole_center), radius=d / 2, length=h), medium=background_material
    )
    
    hole_center
    hole_2 = td.Structure(
        geometry=td.Cylinder(center=(a / 2,  a / 2, hole_center), radius=d / 2, length=h),
        medium=background_material,
    )
    
    
    hole_3 = td.Structure(
        geometry=td.Cylinder(center=(a / 2, -1 * a / 2, hole_center), radius=d / 2, length=h),
        medium=background_material,
    )
    
    
    hole_4 = td.Structure(
        geometry=td.Cylinder(center=(-a / 2,  a / 2, hole_center), radius=d / 2, length=h),
        medium=background_material,
    )
    
    
    hole_5 = td.Structure(
        geometry=td.Cylinder(center=(-a / 2, -1 * a / 2, hole_center), radius=d / 2, length=h),
        medium=background_material,
    )

    sphere = td.Structure(
        geometry=td.Sphere(center=(0, 0, sp-height), radius=d / 2), medium=background_material
    )
    
    # define the base plate structure
    base = td.Structure(
    geometry=td.Box.from_bounds(
        rmin=(-td.inf, -td.inf, sp),  # Lower bound: Bottom of the base
        rmax=(td.inf, td.inf, t_base)          # Upper bound: Top of the base at z = 0
    ),
    medium=material,
    name="MetalBase"
    )
    
    GoldMesh = td.MeshOverrideStructure(
        geometry=td.Box.from_bounds(
        rmin=(-td.inf, -td.inf, sp),  # Lower bound: Bottom of the base
        rmax=(td.inf, td.inf, t_base)          # Upper bound: Top of the base at z = 0
    ),
        dl=(t_base/mesh_res, t_base/mesh_res ,t_base/mesh_res),
        name ='GoldMesh'
)
  
####################

    prism = td.Structure(
    geometry=td.Box.from_bounds(
        rmin=(-td.inf, -td.inf, t_base),  # Lower bound: Bottom of the base
        rmax=(td.inf, td.inf, 100000)          # Upper bound: Top of the base at z = 0
    ),
    medium=prism_material,
    name="Prism"
    )

    sample = td.Structure(
    geometry=td.Box.from_bounds(
        rmin=(-td.inf, -td.inf, -10000000),  # Lower bound: Bottom of the base
        rmax=(td.inf, td.inf, sp)          # Upper bound: Top of the base at z = 0
    ),
    medium=sample_material,
    name="Sample"
    )

    '''    # Combine all holes into a single group
    hole_cell = [hole_1, hole_2, hole_3, hole_4, hole_5]
    holes = td.GeometryGroup(hole_cell)'''
    
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    #allows user to choose Gaussian or plane wave source####################################################################################################################################################################

    #define source
    #plane_wave = td.PlaneWave(
    #source_time= td.GaussianPulse(freq0=freq0, fwidth=fwidth),
    #size=(a, a, 0),
    #center=(0, 0, central_wavelength),
    #direction="+",
    #pol_angle=pol_angle,
    #angle_phi = np.pi/2,
    #angle_theta = np.radians(180-angle),
    #)

    

    #now define the flux monitors
    flux_monitor = td.FluxMonitor(
    center=[0, 0, central_wavelength*1.2], size=[td.inf, td.inf, 0], freqs=freqs, name="R"
    )

    flux_monitor2 = td.FluxMonitor(
    center=[0, 0, -1.2*central_wavelength +t_base], size=[td.inf, td.inf, 0], freqs=freqs, name="T"
    )


    monitor_field = td.FieldMonitor(
    center=[0,0,0],
    size= [0, np.inf,np.inf],
    freqs=freq0,
    name="field",
    )
    
    monitor_field2 = td.FieldMonitor(
    center=[d/2,0,0],
    size= [0, 0,np.inf],
    freqs=freq0,
    name="field2",
    )
    
    #point monitor for field phase
    r_phase_monitor = td.FieldMonitor(
    center=[0,0,central_wavelength*1.2],
    size= [0, 0,0],
    freqs=freq0,
    name="phase",
    )
    
    side = td.FieldMonitor(
    center=[0,0,sp+t_base/2],
    size= [0, np.inf,t_base],
    freqs=freq0,
    name="side",
    )
    top = td.FieldMonitor(
    center=[0,0,sp+h],
    size= [np.inf,np.inf,0],
    freqs=freq0,
    name="top",
    )
    bottom = td.FieldMonitor(
    center=[0,0,sp],
    size= [np.inf,np.inf,0],
    freqs=freq0,
    name="bottom",
    )
###########################################################################################################################################################################################################################
    
    
    def make_sim(angle):
        
        
        
        #define source
        plane_wave = td.PlaneWave(
        source_time= td.GaussianPulse(freq0=freq0, fwidth=fwidth),
        size=(a, a, 0),
        center=(0, 0, central_wavelength),
        direction="+",
        pol_angle=pol_angle,
        angle_phi = np.pi/2,
        angle_theta = np.radians(180-angle),
        )
        
        run_time = run_t / freq0  # simulation run time

        # define simulation domain size
        sim_size = (a,  a,  sim_height*central_wavelength)
        # create the Bloch boundaries
        bloch_x = td.Boundary.bloch_from_source(
        source=plane_wave, domain_size=sim_size[0], axis=0, medium=prism_material
        )
        bloch_y = td.Boundary.bloch_from_source(
        source=plane_wave, domain_size=sim_size[1], axis=1, medium=prism_material
        )

        bspec_bloch = td.BoundarySpec(x=bloch_x, y=bloch_y, z=td.Boundary.pml())
        # Calculate wavevector components and Bloch wavevectors


        grid_spec = td.GridSpec.auto(
        min_steps_per_wvl = grid_res,
        override_structures = [GoldMesh],
        wavelength=lda0,
        )
        
        
            
        if hole == 'one':
            structures=[ base, hole_1, prism, sample]
        elif hole == 'zero':
            structures = [base, prism, sample]
        elif hole == 'five':
            structures = [base,hole_1, hole_2, hole_3, hole_4, hole_5, prism, sample]
        elif hole == 'spherical':
            structures=[base,sphere,prism,sample]

        sim = td.Simulation(
        center=(0, 0, h / 2),
        size=sim_size,
        grid_spec=grid_spec,
        structures = structures,
        sources=[plane_wave],
        monitors=[flux_monitor, monitor_field, flux_monitor2, monitor_field2, r_phase_monitor, side, bottom, top],
        run_time=run_time,
        boundary_spec=bspec_bloch, 
        #symmetry=(1, -1, 0),
        #shutoff=1e-7,  # reducing the default shutoff level
        )
    
        return sim

        #ax = sim.plot(z=h / 2)
        #sim.plot_grid(z=h / 2, ax=ax)

    hole_bot = sp +h
    hole_side = 0
        
        
    
    return make_sim, hole_bot, hole_side, freq0,

def reflectivity(t_data, r_data, freq0):


    #get transmitted flux and evaluate value at only central frequnecy
    trans_flux = t_data.flux
    t = np.abs(trans_flux.sel(f=freq0))

    #the same for reflected
    ref_flux = r_data.flux 
    r = np.abs(ref_flux.sel(f=freq0))
    
    #reflectance 
    i = r + t
    reflectance = (r).item()
    return reflectance

def ref_multi(t_data, r_data, wavelength_range, Nfreqs):
    wr = wavelength_range
    min_wavelength, max_wavelength = wavelength_range
    #frequency and wavelength paramaters
    central_wavelength =np.mean(wavelength_range)
    lda0 = central_wavelength  # central wavelength
    freq0 = td.C_0 / lda0  # central frequency
    ldas = np.linspace(min_wavelength, max_wavelength, Nfreqs)  # wavelength range
    freqs = td.C_0 / ldas  # frequency range

    reflectivities=([])
    for i in range(Nfreqs):

        #get transmitted flux and evaluate value at only central frequnecy
        trans_flux = t_data.flux
        t = np.abs(trans_flux.sel(f=freqs[i]))
    
        #the same for reflected
        ref_flux = r_data.flux 
        r = np.abs(ref_flux.sel(f=freqs[i]))
        
        #reflectance 
        i = r + t
        reflectance = (r).item()
        reflectivities.append(reflectance)
        
    return reflectivities
    
def get_phase(phase_data):

    #Ex = phase_data.Ex
    Ey = phase_data.Ey
    Ez = phase_data.Ez
    E = np.sqrt( Ey**2 + Ez**2)

    phase_y = np.angle(Ey)
    phase_z = np.angle(Ez)
    return phase_y.item(), phase_z.item()

def phase_multi(phase_data, Nfreqs, wavelength_range):

    #Ex = phase_data.Ex
    Ey = phase_data.Ey
    Ez = phase_data.Ez
    E = np.sqrt( Ey**2 + Ez**2)

    phase_y = np.angle(Ey)
    phase_z = np.angle(Ez)
    return phase_y.item(), phase_z.item()
    

def nanohole_scan2(
    wavelength_range,
    Nfreqs,
    source, #choose PlaneWave or Gaussian
    height,#depth of each hole
    spacing,#spacing between holes
    diameter,#hole diamater
    t_base,#metal film thickness
    metal_material,#using JC gold for now
    background_material, #hole material
    prism_material,
    sample_material,
    #angle,
    pol_angle,
    run_t,
    grid_res,
    mesh_res,
    n_prism,
    hole,
    start_angle,
    end_angle,
    steps,
    sim_height):
    
    
    make_sim, hole_bot, hole_side, freq0 = set_nano_sim(wavelength_range,
    Nfreqs,
    source, #choose PlaneWave or Gaussian
    height,#depth of each hole
    spacing,#spacing between holes
    diameter,#hole diamater
    t_base,#metal film thickness
    metal_material,#using JC gold for now
    background_material, #hole material
    prism_material,
    sample_material,
    #angle,
    pol_angle,
    run_t,
    grid_res,
    mesh_res,
    n_prism,
    hole,
    sim_height)
    
    central_wavelength=np.mean(wavelength_range)
    angles = np.linspace(start_angle, end_angle ,steps)

        
    #plot one graph to get pricing and plot ---------------------------------------------------------------------------------------
    nano_sim = make_sim(angle=angles[0])
    print("Plotting for angle", angles[0])
    # visualize geometry
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
    nano_sim.plot(z=hole_bot, ax=ax1)
    nano_sim.plot(y=hole_side, ax=ax2)

    plt.show()
    
    task_id = web.upload(nano_sim, task_name="Nanohole Array")
    estimated_cost = web.estimate_cost(task_id=task_id)
    print("This simulation will cost", estimated_cost, "flex credits")
    print(steps, "simulations will cost", estimated_cost*steps, "flex credits")
    
    input("Proceed?")
    
    
    angles = np.linspace(start_angle, end_angle, steps)
    p_reflectivities = np.zeros(steps)
    phases = np.zeros(steps)
    phases2 = np.zeros(steps)

    sims = {f"l={l:.2f}": make_sim(l) for l in angles}
    batch = web.Batch(simulations=sims, verbose=True)

    batch_results = batch.run(path_dir="data")
    reflectivity_batch = []
    phase_batch_y = []
    phase_batch_z= []
    

    for task_name, sim_data in batch_results.items():
        trans_data = sim_data['T']
        ref_data = sim_data['R']
        phase_data = sim_data['phase']
        
        phases_y, phases_z = get_phase(phase_data)

        #calcualtes array of reflectivities at different wavelenths for each angle point
        reflectivities = ref_multi(trans_data, ref_data, wavelength_range, Nfreqs)

        #print(reflectivities)
            
        reflectivity_batch.append(reflectivities)
        phase_batch_y.append(phases_y)
        phase_batch_z.append(phases_z)

    #print(reflectivity_batch)
      
    phase  = phase_batch_z 
    pz = np.array(phase)
    
    #unwrap the phase for a smooth curve
    uwpz = np.degrees(np.unwrap(pz))

    #interpolate to make a continous curve
    interp_phase = CubicSpline(angles,(uwpz))

    #differentiate the curve 
    phase_div = interp_phase.derivative(1)(angles)


    #calculate the lateral goos-hanchen shift
    A = -2*np.pi / central_wavelength
    gh_shift = (1/A)*phase_div*1e-6
    
    deg_phase_z = np.degrees(phase_batch_z)
    uphase = np.unwrap(deg_phase_z)
    
    deg_phase_y = np.degrees(phase_batch_y)
    uphase_y = np.unwrap(deg_phase_y)
    
    #plt.figure()
    #plt.title("Reflectivity against angles")
    #plt.plot(angles, reflectivity_batch)
    #plt.show()
    
    plt.figure()
    plt.title("Point phase against angles z")
    plt.plot(angles, phase_batch_z)
    plt.show()
    
    plt.figure()
    plt.title("Lateral shift agains angles")
    plt.plot(angles, gh_shift)
    plt.show()

    ref = reflectivity_batch

    wavelength_max = np.max(wavelength_range)
    wavelength_min = np.min(wavelength_range)
    wavelengths = np.linspace(wavelength_min, wavelength_max, Nfreqs)
    
    # ref is angle x wavelength
    # ref[i][j]: i-th angle, j-th wavelength
    
    plt.figure(figsize=(8,6))
    for j in range(Nfreqs):
        # Extract the j-th wavelength data across all angles
        wavelength_reflectivity = [ref[i][j] for i in range(len(angles))]
        
        # Plot reflectivity vs angle for this wavelength
        plt.plot(angles, wavelength_reflectivity, label=f"{wavelengths[j]} nm")
    
    plt.xlabel("Angle (degrees)")
    plt.ylabel("Reflectivity")
    plt.title("Reflectivity vs Angle for Different Wavelengths")
    plt.grid(True)
    plt.legend()
    plt.show()



    return phase, gh_shift, reflectivity_batch, angles


def nanohole_scan(
    wavelength_range,
    Nfreqs,
    source, #choose PlaneWave or Gaussian
    height,#depth of each hole
    spacing,#spacing between holes
    diameter,#hole diamater
    t_base,#metal film thickness
    metal_material,#using JC gold for now
    background_material, #hole material
    prism_material,
    sample_material,
    #angle,
    pol_angle,
    run_t,
    grid_res,
    mesh_res,
    n_prism,
    hole,
    start_angle,
    end_angle,
    steps,
    sim_height):
    
    
    make_sim, hole_bot, hole_side, freq0 = set_nano_sim(wavelength_range,
    Nfreqs,
    source, #choose PlaneWave or Gaussian
    height,#depth of each hole
    spacing,#spacing between holes
    diameter,#hole diamater
    t_base,#metal film thickness
    metal_material,#using JC gold for now
    background_material, #hole material
    prism_material,
    sample_material,
    #angle,
    pol_angle,
    run_t,
    grid_res,
    mesh_res,
    n_prism,
    hole,
    sim_height)
    
    central_wavelength=np.mean(wavelength_range)
    angles = np.linspace(start_angle, end_angle ,steps)

        
    #plot one graph to get pricing and plot ---------------------------------------------------------------------------------------
    nano_sim = make_sim(angle=angles[0])
    print("Plotting for angle", angles[0])
    # visualize geometry
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
    nano_sim.plot(z=hole_bot, ax=ax1)
    nano_sim.plot(y=hole_side, ax=ax2)

    plt.show()
    
    task_id = web.upload(nano_sim, task_name="Nanohole Array")
    estimated_cost = web.estimate_cost(task_id=task_id)
    print("This simulation will cost", estimated_cost, "flex credits")
    print(steps, "simulations will cost", estimated_cost*steps, "flex credits")
    
    input("Proceed?")
    
    
    angles = np.linspace(start_angle, end_angle, steps)
    p_reflectivities = np.zeros(steps)
    phases = np.zeros(steps)
    phases2 = np.zeros(steps)

    sims = {f"l={l:.2f}": make_sim(l) for l in angles}
    batch = web.Batch(simulations=sims, verbose=True)

    batch_results = batch.run(path_dir="data")
    reflectivity_batch = []
    phase_batch_y = []
    phase_batch_z= []
    

    for task_name, sim_data in batch_results.items():
        trans_data = sim_data['T']
        ref_data = sim_data['R']
        phase_data = sim_data['phase']
        
        phases_y, phases_z = get_phase(phase_data)

        #calcualtes array of reflectivities at different wavelenths for each angle point
        reflectivities = reflectivity(trans_data, ref_data, freq0)

        #print(reflectivities)
            
        reflectivity_batch.append(reflectivities)
        phase_batch_y.append(phases_y)
        phase_batch_z.append(phases_z)

    #print(reflectivity_batch)
    ref = reflectivity_batch
    phasez  = phase_batch_z 
    phase = np.array(phasez)
    pz = phase
    
    #unwrap the phase for a smooth curve
    uwpz = np.degrees(np.unwrap(pz))

    #interpolate to make a continous curve
    interp_phase = CubicSpline(angles,(uwpz))

    #differentiate the curve 
    phase_div = interp_phase.derivative(1)(angles)


    #calculate the lateral goos-hanchen shift
    A = -2*np.pi / central_wavelength
    gh_shift = (1/A)*phase_div*1e-6
    
    deg_phase_z = np.degrees(phase_batch_z)
    uphase = np.unwrap(deg_phase_z)
    
    deg_phase_y = np.degrees(phase_batch_y)
    uphase_y = np.unwrap(deg_phase_y)
    
    #plt.figure()
    #plt.title("Reflectivity against angles")
    #plt.plot(angles, reflectivity_batch)
    #plt.show()
    '''
    plt.figure()
    plt.title("Point phase against angles z")
    plt.plot(angles, phase_batch_z)
    plt.show()
    
    plt.figure()
    plt.title("Lateral shift agains angles")
    plt.plot(angles, gh_shift)
    plt.show()

    ref = reflectivity_batch

    wavelength_max = np.max(wavelength_range)
    wavelength_min = np.min(wavelength_range)
    wavelengths = np.linspace(wavelength_min, wavelength_max, Nfreqs)
    
    # ref is angle x wavelength
    # ref[i][j]: i-th angle, j-th wavelength
    
    plt.figure()
    plt.plot(angles, ref)
    
    plt.xlabel("Angle (degrees)")
    plt.ylabel("Reflectivity")
    plt.title("Reflectivity vs Angle for Different Wavelengths")
    plt.grid(True)
    plt.legend()
    plt.show()
    '''

    return phase, gh_shift, ref, angles
                              
                             
def plot_compare(d1, d2, start_x, end_x, steps, steps2, title, label1, label2):
    plt.figure()
    plt.title(title)
    array1 = np.linspace(start_x, end_x, steps)
    array2 = np.linspace(start_x, end_x, steps2)
    plt.plot(array, d1, label=label1)
    plt.plot(array2, d2, label=label2)
    plt.show()
    
def figsave(angles, values, title, x_label, y_label, folder='investigations'):
    """
    Save data and plot into a specified folder.

    Parameters
    ----------
    angles : array-like
        The angle values for the x-axis.
    values : array-like
        The corresponding reflectivity (or other) values for the y-axis.
    title : str
        The title of the plot (will also be used as part of the filename).
    x_label : str
        The label for the x-axis.
    y_label : str
        The label for the y-axis.
    folder : str, optional
        The folder name where the files will be saved (default is 'investigations').

    Returns
    -------
    None
    """
    # Ensure the output directory exists
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    # Convert data to a DataFrame for easy CSV export
    df = pd.DataFrame({
        x_label: angles,
        y_label: values
    })
    
    # Create a filename-friendly version of the title by replacing spaces
    filename_base = title.replace(" ", "_")
    
    # Save the DataFrame to CSV
    csv_path = os.path.join(folder, f"{filename_base}.csv")
    df.to_csv(csv_path, index=False)
    
    # Plot the data
    plt.figure(figsize=(8, 6))
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.plot(angles, values)
    
    # Save the plot as a PDF
    pdf_path = os.path.join(folder, f"{filename_base}.pdf")
    plt.savefig(pdf_path)
    plt.show()
    plt.close()    
                                 
def get_intensity(data):
    Ex = data.Ex 
    Ey = data.Ey
    Ez = data.Ez
    E = np.sqrt(Ex**2 + Ey**2 + Ez**2)
    Int = np.abs(E)**2

    return Int
    
def figsave2(d1, d2, start_x, end_x, steps, steps2, title, label1, label2, folder='investigations'):
    """
    Plot two datasets for comparison, save the data and the plot.
    
    Parameters
    ----------
    d1 : array-like
        The first set of values to plot.
    d2 : array-like
        The second set of values to plot.
    start_x : float
        The start value of the x-range.
    end_x : float
        The end value of the x-range.
    steps : int
        The number of steps (points) for the first dataset.
    steps2 : int
        The number of steps (points) for the second dataset.
    title : str
        The title of the plot (will also be used as part of the filename).
    label1 : str
        The label for the first dataset in the legend.
    label2 : str
        The label for the second dataset in the legend.
    folder : str, optional
        The folder name where the files will be saved (default is 'investigations').

    Returns
    -------
    None
    """
    # Ensure the output directory exists
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    # Create the x arrays for both datasets
    x1 = np.linspace(start_x, end_x, steps)
    x2 = np.linspace(start_x, end_x, steps2)

    # Create a DataFrame that holds both datasets
    # We'll have columns: X1, D1, X2, D2
    df = pd.DataFrame({
        "X1": x1,
        label1: d1,
        "X2": x2,
        label2: d2
    })

    # Create a filename-friendly version of the title
    filename_base = title.replace(" ", "_")
    
    # Save the data as a CSV
    csv_path = os.path.join(folder, f"{filename_base}.csv")
    df.to_csv(csv_path, index=False)
    
    # Plot the data
    plt.figure(figsize=(8,6))
    plt.title(title)
    plt.plot(x1, d1, label=label1)
    plt.plot(x2, d2, label=label2)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()

    # Save the plot as a PDF
    pdf_path = os.path.join(folder, f"{filename_base}.pdf")
    plt.savefig(pdf_path)

    # Now show the plot
    plt.show()
    
    # Close the figure
    plt.close()

#remake with new multi wavelength
def sens_scan(
    wavelength_range,
    Nfreqs,
    source, #choose PlaneWave or Gaussian
    height,#depth of each hole
    spacing,#spacing between holes
    diameter,#hole diamater
    t_base,#metal film thickness
    metal_material,#using JC gold for now
     #hole material
    prism_material,
    start_sample,
    end_sample,
    sample_step,
    #angle,
    pol_angle,
    run_t,
    grid_res,
    mesh_res,
    n_prism,
    hole,
    start_angle,
    end_angle,
    steps,
    sim_height):
    
    indices = np.linspace(start_sample, end_sample, sample_step)
    
    make_sim, hole_bot, hole_side, freq0 = set_nano_sim(wavelength_range,
    Nfreqs,
    source, #choose PlaneWave or Gaussian
    height,#depth of each hole
    spacing,#spacing between holes
    diameter,#hole diamater
    t_base,#metal film thickness
    metal_material,#using JC gold for now
    background_material, #hole material
    prism_material,
    sample_material,
    #angle,
    pol_angle,
    run_t,
    grid_res,
    mesh_res,
    n_prism,
    hole,
    sim_height)
    
    central_wavelength=np.mean(wavelength_range)
    angles = np.linspace(start_angle, end_angle ,steps)

        
    #plot one graph to get pricing and plot ---------------------------------------------------------------------------------------
    nano_sim = make_sim(angle=angles[0])
    print("Plotting for angle", angles[0])
    # visualize geometry
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
    nano_sim.plot(z=hole_bot, ax=ax1)
    nano_sim.plot(y=hole_side, ax=ax2)

    plt.show()
    
    task_id = web.upload(nano_sim, task_name="Nanohole Array")
    estimated_cost = web.estimate_cost(task_id=task_id)
    print("This simulation will cost", estimated_cost, "flex credits")
    print(steps, "simulations will cost", estimated_cost*steps, "flex credits")
    
    input("Proceed?")
    
    
    angles = np.linspace(start_angle, end_angle, steps)
    p_reflectivities = np.zeros(steps)
    phases = np.zeros(steps)
    phases2 = np.zeros(steps)

    sims = {f"l={l:.2f}": make_sim(l) for l in angles}
    batch = web.Batch(simulations=sims, verbose=True)

    batch_results = batch.run(path_dir="data")
    reflectivity_batch = []
    phase_batch_y = []
    phase_batch_z= []
    

    for task_name, sim_data in batch_results.items():
        trans_data = sim_data['T']
        ref_data = sim_data['R']
        phase_data = sim_data['phase']
        coupling = sim_data['side']

        
        
        phases_y, phases_z = get_phase(phase_data)
        reflectivities = ref_multi(trans_data, ref_data, freq0)
        reflectivity_batch.append(reflectivities)
        phase_batch_y.append(phases_y)
        phase_batch_z.append(phases_z)
      
    phase  = phase_batch_z 
    pz = np.array(phase)
    
    #unwrap the phase for a smooth curve
    uwpz = np.degrees(np.unwrap(pz))

    #interpolate to make a continous curve
    interp_phase = CubicSpline(angles,(uwpz))

    #differentiate the curve 
    phase_div = interp_phase.derivative(1)(angles)


    #calculate the lateral goos-hanchen shift
    A = -2*np.pi / central_wavelength
    gh_shift = (1/A)*phase_div*1e-6
    
    deg_phase_z = np.degrees(phase_batch_z)
    uphase = np.unwrap(deg_phase_z)
    
    deg_phase_y = np.degrees(phase_batch_y)
    uphase_y = np.unwrap(deg_phase_y)
    
    plt.figure()
    plt.title("Reflectivity against angles")
    plt.plot(angles, reflectivity_batch)
    plt.show()
    
    plt.figure()
    plt.title("Point phase against angles z")
    plt.plot(angles, phase_batch_z)
    plt.show()
    
    plt.figure()
    plt.title("Lateral shift agains angles")
    plt.plot(angles, gh_shift)
    plt.show()



    return phase, gh_shift, reflectivity_batch, angles

#BELOW IS THE POROUS FUNCTION CREATION 

def create_ellipsoid(a, b, c, u_segments, v_segments):
    # initialize empty lists for vertices and faces
    vertices = []
    faces = []

    # create vertices
    for i in range(u_segments + 1):
        theta = i * np.pi / u_segments  # angle for the latitude (0 to pi)

        for j in range(v_segments + 1):
            phi = j * 2 * np.pi / v_segments  # angle for the longitude (0 to 2*pi)

            # compute vertex position using ellipsoidal equations
            x = a * np.sin(theta) * np.cos(phi)
            y = b * np.sin(theta) * np.sin(phi)
            z = c * np.cos(theta)

            vertices.append([x, y, z])

    # create faces
    for i in range(u_segments):
        for j in range(v_segments):
            # compute indices for vertices
            v1 = i * (v_segments + 1) + j
            v2 = (i + 1) * (v_segments + 1) + j
            v3 = (i + 1) * (v_segments + 1) + (j + 1)
            v4 = i * (v_segments + 1) + (j + 1)

            # create faces using the vertices
            faces.append([v1, v2, v3])
            faces.append([v1, v3, v4])

    # create mesh using the generated vertices and faces
    return trimesh.Trimesh(vertices=vertices, faces=faces)


def rand_coord(low_y, high_y,low_z,high_z, particles):
    #will generate random lateral coordinates between range
    #will generate random z coords make sure this is within film range
    
    N = particles  # number of random points
    x_coords = np.random.uniform(low_y, high_y, size=N)
    y_coords = np.random.uniform(low_y, high_y, size=N)
    z_coords = np.random.uniform(low_z, high_z, size=N)
    
    points = np.column_stack((x_coords, y_coords, z_coords))
    return(points)


def rand_ellipse(a_low, a_high, b_low, b_high, c_low, c_high, particles, u_segments, v_segments):
    ellipsoids = []
    for _ in range(particles):
        # Generate random values for a, b, c
        a = np.random.uniform(a_low, a_high)
        b = np.random.uniform(b_low, b_high)
        c = np.random.uniform(c_low, c_high)

        # Create the ellipsoid and add it to the list
        ellipsoid_mesh = create_ellipsoid(a, b, c, u_segments, v_segments)
        ellipsoids.append(ellipsoid_mesh)

    return ellipsoids
        
def pore_structure_rand(x_range, y_range, z_range,length_range, height_range, thick_range, pores, pore_material):
    x_low, x_high = x_range
    y_low, y_high = y_range
    z_low, z_high = z_range
    a_low, a_high = length_range
    b_low, b_high = height_range
    c_low, c_high = thick_range

    #defines resolution of ellipsoid creation 100 is good generally
    u_segments = v_segments = 100

    #generates random coords on grids for pores to take
    coords = rand_coord(y_low, y_high,z_low,z_high, pores)

    ellipses = rand_ellipse(a_low, a_high, b_low, b_high, c_low, c_high, pores, u_segments, v_segments)

    strucks=[]
    for i in range(pores):
        #define random rotation transform
        #need a transform about y (0,1,0)
        #then z z(0,0,1) for safety can also x
        x_val = np.random.uniform(0,2*np.pi)
        y_val = np.random.uniform(0,2*np.pi)
        z_val = np.random.uniform(0,2*np.pi)

        #make sure to rotate them about their own centres so they remain in film
        x_rot = trimesh.transformations.rotation_matrix(x_val, [coords[i][0],0,0])
        y_rot = trimesh.transformations.rotation_matrix(y_val, [0,coords[i][1],0])
        z_rot = trimesh.transformations.rotation_matrix(z_val, [0,0,coords[i][2]])
        #dissalowed rotation for now will fix param to height allowing confinement in metal
        
        ellipses[i].apply_translation(coords[i])
        ellipses[i].apply_transform(x_rot)
        ellipses[i].apply_transform(y_rot)
        ellipses[i].apply_transform(z_rot)
    
    
        
    
        ellipse = td.TriangleMesh.from_trimesh(ellipses[i])
        ellipse_str = td.Structure(geometry=ellipse, medium=pore_material)
        strucks.append(ellipse_str)
    
    listey=[]
    for i in range(len(strucks)):
        listey.append(strucks[i])

    return(listey)










#PORE SIM FUNC###############################################################################################################################
def set_pore_sim(
    wavelength_range,
    Nfreqs,
    source, #choose PlaneWave or Gaussian
    height,#depth of each hole
    spacing,#spacing between holes
    diameter,#hole diamater
    t_base,#metal film thickness
    metal_material,#using JC gold for now
    background_material, #hole material
    prism_material,
    sample_material,
    #angle,
    pol_angle,
    run_t,
    grid_res,
    mesh_res,
    n_prism,
    hole,
    sim_height,
    length_range, 
    height_range,
    thick_range,
    pores):

    #SET PARAMATERS 
    
    #n_prism = np.sqrt(prism_material)
    stack_point =0
    sp = stack_point
    
    
    min_wavelength, max_wavelength = wavelength_range
    #frequency and wavelength paramaters
    central_wavelength =np.mean(wavelength_range)
    lda0 = central_wavelength  # central wavelength
    freq0 = td.C_0 / lda0  # central frequency
    ldas = np.linspace(min_wavelength, max_wavelength, Nfreqs)  # wavelength range
    freqs = td.C_0 / ldas  # frequency range
    fwidth = 0.5 * (np.max(freqs) - np.min(freqs))  # width of the source frequency range

    #-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #define geometry for nanohole array
    h = height #depth of each hole 
    a = spacing #spacing between hole 
    d = diameter #diamater of hole
    t_base = t_base # thickness of base film

    material = metal_material
    hole_center = sp +h/2
    #CREATE UP TO FIVE HOLES FOR A UNIT CELL

    #here is the issue btw-----------------------------------------------------------------------------------------------------
    z_range =(sp,t_base)
              
    print(z_range)
    
    #define the pore structure
    pores = pore_structure_rand(x_range=(-a/2,a/2), y_range =(-a/2,a/2), z_range=z_range,
                    length_range=length_range, height_range=height_range, thick_range=thick_range, pores=pores
                    , pore_material = sample_material)

   
    hole_1 = td.Structure(
        geometry=td.Cylinder(center=(0, 0, hole_center), radius=d / 2, length=h), medium=background_material
    )
    
    hole_center
    hole_2 = td.Structure(
        geometry=td.Cylinder(center=(a / 2,  a / 2, hole_center), radius=d / 2, length=h),
        medium=background_material,
    )
    
    
    hole_3 = td.Structure(
        geometry=td.Cylinder(center=(a / 2, -1 * a / 2, hole_center), radius=d / 2, length=h),
        medium=background_material,
    )
    
    
    hole_4 = td.Structure(
        geometry=td.Cylinder(center=(-a / 2,  a / 2, hole_center), radius=d / 2, length=h),
        medium=background_material,
    )
    
    
    hole_5 = td.Structure(
        geometry=td.Cylinder(center=(-a / 2, -1 * a / 2, hole_center), radius=d / 2, length=h),
        medium=background_material,
    )

    sphere = td.Structure(
        geometry=td.Sphere(center=(0, 0, sp-height), radius=d / 2), medium=background_material
    )
    
    # define the base plate structure
    base = td.Structure(
    geometry=td.Box.from_bounds(
        rmin=(-td.inf, -td.inf, sp),  # Lower bound: Bottom of the base
        rmax=(td.inf, td.inf, t_base)          # Upper bound: Top of the base at z = 0
    ),
    medium=material,
    name="MetalBase"
    )

    length_min = np.min(length_range)
    height_min = np.min(height_range)
    thick_min = np.min(thick_range)
    
    GoldMesh = td.MeshOverrideStructure(
        geometry=td.Box.from_bounds(
        rmin=(-td.inf, -td.inf, sp),  # Lower bound: Bottom of the base
        rmax=(td.inf, td.inf, t_base)          # Upper bound: Top of the base at z = 0
    ),
        dl=(length_min/mesh_res, thick_min/mesh_res ,height_min/mesh_res),
        name ='GoldMesh'
)
  
####################

    prism = td.Structure(
    geometry=td.Box.from_bounds(
        rmin=(-td.inf, -td.inf, t_base),  # Lower bound: Bottom of the base
        rmax=(td.inf, td.inf, 100000)          # Upper bound: Top of the base at z = 0
    ),
    medium=prism_material,
    name="Prism"
    )

    sample = td.Structure(
    geometry=td.Box.from_bounds(
        rmin=(-td.inf, -td.inf, -10000000),  # Lower bound: Bottom of the base
        rmax=(td.inf, td.inf, sp)          # Upper bound: Top of the base at z = 0
    ),
    medium=sample_material,
    name="Sample"
    )

    '''    # Combine all holes into a single group
    hole_cell = [hole_1, hole_2, hole_3, hole_4, hole_5]
    holes = td.GeometryGroup(hole_cell)'''
    
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    #allows user to choose Gaussian or plane wave source####################################################################################################################################################################

    #define source
    #plane_wave = td.PlaneWave(
    #source_time= td.GaussianPulse(freq0=freq0, fwidth=fwidth),
    #size=(a, a, 0),
    #center=(0, 0, central_wavelength),
    #direction="+",
    #pol_angle=pol_angle,
    #angle_phi = np.pi/2,
    #angle_theta = np.radians(180-angle),
    #)

    

    #now define the flux monitors
    flux_monitor = td.FluxMonitor(
    center=[0, 0, central_wavelength*1.2], size=[td.inf, td.inf, 0], freqs=freqs, name="R"
    )

    flux_monitor2 = td.FluxMonitor(
    center=[0, 0, -1.2*central_wavelength +t_base], size=[td.inf, td.inf, 0], freqs=freqs, name="T"
    )


    monitor_field = td.FieldMonitor(
    center=[0,0,0],
    size= [0, np.inf,np.inf],
    freqs=freq0,
    name="field",
    )
    
    monitor_field2 = td.FieldMonitor(
    center=[d/2,0,0],
    size= [0, 0,np.inf],
    freqs=freq0,
    name="field2",
    )
    
    #point monitor for field phase
    r_phase_monitor = td.FieldMonitor(
    center=[0,0,central_wavelength*1.2],
    size= [0, 0,0],
    freqs=freq0,
    name="phase",
    )
    
    side = td.FieldMonitor(
    center=[0,0,sp+t_base/2],
    size= [0, np.inf,t_base],
    freqs=freq0,
    name="side",
    )
    top = td.FieldMonitor(
    center=[0,0,sp+h],
    size= [np.inf,np.inf,0],
    freqs=freq0,
    name="top",
    )
    bottom = td.FieldMonitor(
    center=[0,0,sp],
    size= [np.inf,np.inf,0],
    freqs=freq0,
    name="bottom",
    )
###########################################################################################################################################################################################################################
    
    
    def make_sim(angle):
        
        
        
        #define source
        plane_wave = td.PlaneWave(
        source_time= td.GaussianPulse(freq0=freq0, fwidth=fwidth),
        size=(a, a, 0),
        center=(0, 0, central_wavelength),
        direction="+",
        pol_angle=pol_angle,
        angle_phi = np.pi/2,
        angle_theta = np.radians(180-angle),
        )
        
        run_time = run_t / freq0  # simulation run time

        # define simulation domain size
        sim_size = (a,  a,  sim_height*central_wavelength)
        # create the Bloch boundaries
        bloch_x = td.Boundary.bloch_from_source(
        source=plane_wave, domain_size=sim_size[0], axis=0, medium=prism_material
        )
        bloch_y = td.Boundary.bloch_from_source(
        source=plane_wave, domain_size=sim_size[1], axis=1, medium=prism_material
        )

        bspec_bloch = td.BoundarySpec(x=bloch_x, y=bloch_y, z=td.Boundary.pml())
        # Calculate wavevector components and Bloch wavevectors


        grid_spec = td.GridSpec.auto(
        min_steps_per_wvl = grid_res,
        override_structures = [GoldMesh],
        wavelength=lda0,
        )
        
        
            
        if hole == 'one':
            structures=[ base, hole_1, prism, sample]
        elif hole == 'zero':
            structures = [base, prism, sample]
        elif hole == 'five':
            structures = [base,hole_1, hole_2, hole_3, hole_4, hole_5, prism, sample]
        elif hole == 'spherical':
            structures=[base,sphere,prism,sample]
        elif hole=='pores':
            
            structures =[base,sample] + pores + [prism] #[base] + pores + [prism,sample]
            #structures.append(base)
            #structures.append(prism)
            #structures.append(sample)

        
        sim = td.Simulation(
        center=(0, 0, 0),
        size=sim_size,
        grid_spec=grid_spec,
        structures = structures,
        sources=[plane_wave],
        monitors=[flux_monitor, monitor_field, flux_monitor2, monitor_field2, r_phase_monitor, side, bottom, top],
        run_time=run_time,
        boundary_spec=bspec_bloch, 
        #symmetry=(1, -1, 0),
        #shutoff=1e-7,  # reducing the default shutoff level
        )
    
        return sim

        #ax = sim.plot(z=h / 2)
        #sim.plot_grid(z=h / 2, ax=ax)

    hole_bot = sp +h
    hole_side = 0
        
        
    
    return make_sim, hole_bot, hole_side, freq0,

def pore_scan(
    wavelength_range,
    Nfreqs,
    source, #choose PlaneWave or Gaussian
    height,#depth of each hole
    spacing,#spacing between holes
    diameter,#hole diamater
    t_base,#metal film thickness
    metal_material,#using JC gold for now
    background_material, #hole material
    prism_material,
    sample_material,
    #angle,
    pol_angle,
    run_t,
    grid_res,
    mesh_res,
    n_prism,
    hole,
    start_angle,
    end_angle,
    steps,
    sim_height,
    length_range,
    height_range,
    thick_range,
    pores):
    
    
    make_sim, hole_bot, hole_side, freq0 = set_pore_sim(wavelength_range,
    Nfreqs,
    source, #choose PlaneWave or Gaussian
    height,#depth of each hole
    spacing,#spacing between holes
    diameter,#hole diamater
    t_base,#metal film thickness
    metal_material,#using JC gold for now
    background_material, #hole material
    prism_material,
    sample_material,
    #angle,
    pol_angle,
    run_t,
    grid_res,
    mesh_res,
    n_prism,
    hole,
    sim_height,
    length_range,
    height_range,
    thick_range,
    pores)
    
    central_wavelength=np.mean(wavelength_range)
    angles = np.linspace(start_angle, end_angle ,steps)

        
    #plot one graph to get pricing and plot ---------------------------------------------------------------------------------------
    nano_sim = make_sim(angle=angles[0])
    print("Plotting for angle", angles[0])
    # visualize geometry
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    # Top-Left Subplot: Z-axis slices
    axs[0, 0].set_title('Z-axis Slices')
    nano_sim.plot(z=hole_bot, ax=axs[0, 0], label='Hole Bottom')
    nano_sim.plot(z=t_base, ax=axs[0, 0], label='T Base')
    axs[0, 0].legend()
    axs[0, 0].set_xlabel('X-axis')
    axs[0, 0].set_ylabel('Y-axis')
    
    # Top-Right Subplot: Y-axis slice at hole_side
    axs[0, 1].set_title('Y-axis Slice')
    nano_sim.plot(y=hole_side,  ax=axs[0, 1], label='Hole Side')
    axs[0, 1].legend()
    axs[0, 1].set_xlabel('X-axis')
    axs[0, 1].set_ylabel('Z-axis')
    axs[0,1].set_ylim(0,t_base)
    
    # Bottom-Left Subplot: X=0 Plane
    axs[1, 0].set_title('X=0 Plane')
    nano_sim.plot(x=0, ax=axs[1, 0], label='X=0 Slice')
    axs[1, 0].legend()
    axs[1, 0].set_xlabel('Y-axis')
    axs[1, 0].set_ylabel('Z-axis')
    axs[1,0].set_ylim(0,t_base)
    
    # Bottom-Right Subplot: Y=0 Plane
    axs[1, 1].set_title('Y=0 Plane')
    nano_sim.plot(y=0, ax=axs[1, 1], label='Y=0 Slice')
    axs[1, 1].legend()
    axs[1, 1].set_xlabel('X-axis')
    axs[1, 1].set_ylabel('Z-axis')
    
    # Adjust layout for better spacing
    plt.tight_layout()
        #nano_sim.plot_3d()
    
    plt.show()
    
    task_id = web.upload(nano_sim, task_name="Nanohole Array")
    estimated_cost = web.estimate_cost(task_id=task_id)
    print("This simulation will cost", estimated_cost, "flex credits")
    print(steps, "simulations will cost", estimated_cost*steps, "flex credits")
    
    input("Proceed?")
    
    
    angles = np.linspace(start_angle, end_angle, steps)
    p_reflectivities = np.zeros(steps)
    phases = np.zeros(steps)
    phases2 = np.zeros(steps)

    sims = {f"l={l:.2f}": make_sim(l) for l in angles}
    batch = web.Batch(simulations=sims, verbose=True)

    batch_results = batch.run(path_dir="data")
    reflectivity_batch = []
    phase_batch_y = []
    phase_batch_z= []
    

    for task_name, sim_data in batch_results.items():
        trans_data = sim_data['T']
        ref_data = sim_data['R']
        phase_data = sim_data['phase']
        
        phases_y, phases_z = get_phase(phase_data)

        #calcualtes array of reflectivities at different wavelenths for each angle point
        reflectivities = reflectivity(trans_data, ref_data, freq0)

        #print(reflectivities)
            
        reflectivity_batch.append(reflectivities)
        phase_batch_y.append(phases_y)
        phase_batch_z.append(phases_z)

    #print(reflectivity_batch)
    ref = reflectivity_batch
    phasez  = phase_batch_z 
    phase = np.array(phasez)
    pz = phase
    
    #unwrap the phase for a smooth curve
    uwpz = np.degrees(np.unwrap(pz))

    #interpolate to make a continous curve
    interp_phase = CubicSpline(angles,(uwpz))

    #differentiate the curve 
    phase_div = interp_phase.derivative(1)(angles)


    #calculate the lateral goos-hanchen shift
    A = -2*np.pi / central_wavelength
    gh_shift = (1/A)*phase_div*1e-6
    
    deg_phase_z = np.degrees(phase_batch_z)
    uphase = np.unwrap(deg_phase_z)
    
    deg_phase_y = np.degrees(phase_batch_y)
    uphase_y = np.unwrap(deg_phase_y)
    
    #plt.figure()
    #plt.title("Reflectivity against angles")
    #plt.plot(angles, reflectivity_batch)
    #plt.show()
    '''
    plt.figure()
    plt.title("Point phase against angles z")
    plt.plot(angles, phase_batch_z)
    plt.show()
    
    plt.figure()
    plt.title("Lateral shift agains angles")
    plt.plot(angles, gh_shift)
    plt.show()

    ref = reflectivity_batch

    wavelength_max = np.max(wavelength_range)
    wavelength_min = np.min(wavelength_range)
    wavelengths = np.linspace(wavelength_min, wavelength_max, Nfreqs)
    
    # ref is angle x wavelength
    # ref[i][j]: i-th angle, j-th wavelength
    
    plt.figure()
    plt.plot(angles, ref)
    
    plt.xlabel("Angle (degrees)")
    plt.ylabel("Reflectivity")
    plt.title("Reflectivity vs Angle for Different Wavelengths")
    plt.grid(True)
    plt.legend()
    plt.show()
    '''

    return phase, gh_shift, ref, angles



#DELEte BELOW HONESTLY I HATE IT

########################################################################################################################################################
#PORE SIM FUNC###############################################################################################################################
def set_pore_sim2(
    wavelength_range,
    Nfreqs,
    source, #choose PlaneWave or Gaussian
    height,#depth of each hole
    spacing,#spacing between holes
    diameter,#hole diamater
    t_base,#metal film thickness
    metal_material,#using JC gold for now
    background_material, #hole material
    prism_material,
    sample_material,
    #angle,
    pol_angle,
    run_t,
    grid_res,
    mesh_res,
    n_prism,
    hole,
    sim_height,
    length_range, 
    height_range,
    thick_range,
    pores):

    #SET PARAMATERS 
    
    #n_prism = np.sqrt(prism_material)
    stack_point =0
    sp = stack_point
    
    
    min_wavelength, max_wavelength = wavelength_range
    #frequency and wavelength paramaters
    central_wavelength =np.mean(wavelength_range)
    lda0 = central_wavelength  # central wavelength
    freq0 = td.C_0 / lda0  # central frequency
    ldas = np.linspace(min_wavelength, max_wavelength, Nfreqs)  # wavelength range
    freqs = td.C_0 / ldas  # frequency range
    fwidth = 0.5 * (np.max(freqs) - np.min(freqs))  # width of the source frequency range

    #-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #define geometry for nanohole array
    h = height #depth of each hole 
    a = spacing #spacing between hole 
    d = diameter #diamater of hole
    t_base = t_base # thickness of base film

    material = metal_material
    hole_center = sp +h/2
    #CREATE UP TO FIVE HOLES FOR A UNIT CELL

    #here is the issue btw-----------------------------------------------------------------------------------------------------
    z_range =(sp,t_base- (np.max(thick_range)/2))
              
    print(z_range)
    
    #define the pore structure
    pores = pore_structure_rand(x_range=(-a/2,a/2), y_range =(-a/2,a/2), z_range=z_range,
                    length_range=length_range, height_range=height_range, thick_range=thick_range, pores=pores
                    , pore_material = sample_material)

   
    hole_1 = td.Structure(
        geometry=td.Cylinder(center=(0, 0, hole_center), radius=d / 2, length=h), medium=background_material
    )
    
    hole_center
    hole_2 = td.Structure(
        geometry=td.Cylinder(center=(a / 2,  a / 2, hole_center), radius=d / 2, length=h),
        medium=background_material,
    )
    
    
    hole_3 = td.Structure(
        geometry=td.Cylinder(center=(a / 2, -1 * a / 2, hole_center), radius=d / 2, length=h),
        medium=background_material,
    )
    
    
    hole_4 = td.Structure(
        geometry=td.Cylinder(center=(-a / 2,  a / 2, hole_center), radius=d / 2, length=h),
        medium=background_material,
    )
    
    
    hole_5 = td.Structure(
        geometry=td.Cylinder(center=(-a / 2, -1 * a / 2, hole_center), radius=d / 2, length=h),
        medium=background_material,
    )

    sphere = td.Structure(
        geometry=td.Sphere(center=(0, 0, sp-height), radius=d / 2), medium=background_material
    )
    
    # define the base plate structure
    base = td.Structure(
    geometry=td.Box.from_bounds(
        rmin=(-td.inf, -td.inf, sp),  # Lower bound: Bottom of the base
        rmax=(td.inf, td.inf, t_base)          # Upper bound: Top of the base at z = 0
    ),
    medium=material,
    name="MetalBase"
    )

    length_min = np.min(length_range)
    height_min = np.min(height_range)
    thick_min = np.min(thick_range)
    
    GoldMesh = td.MeshOverrideStructure(
        geometry=td.Box.from_bounds(
        rmin=(-td.inf, -td.inf, sp),  # Lower bound: Bottom of the base
        rmax=(td.inf, td.inf, t_base)          # Upper bound: Top of the base at z = 0
    ),
        dl=(length_min/mesh_res, thick_min/mesh_res ,height_min/mesh_res),
        name ='GoldMesh'
)
  
####################

    prism = td.Structure(
    geometry=td.Box.from_bounds(
        rmin=(-td.inf, -td.inf, t_base),  # Lower bound: Bottom of the base
        rmax=(td.inf, td.inf, 100000)          # Upper bound: Top of the base at z = 0
    ),
    medium=prism_material,
    name="Prism"
    )

    sample = td.Structure(
    geometry=td.Box.from_bounds(
        rmin=(-td.inf, -td.inf, -10000000),  # Lower bound: Bottom of the base
        rmax=(td.inf, td.inf, sp)          # Upper bound: Top of the base at z = 0
    ),
    medium=sample_material,
    name="Sample"
    )

    '''    # Combine all holes into a single group
    hole_cell = [hole_1, hole_2, hole_3, hole_4, hole_5]
    holes = td.GeometryGroup(hole_cell)'''
    
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    #allows user to choose Gaussian or plane wave source####################################################################################################################################################################

    #define source
    #plane_wave = td.PlaneWave(
    #source_time= td.GaussianPulse(freq0=freq0, fwidth=fwidth),
    #size=(a, a, 0),
    #center=(0, 0, central_wavelength),
    #direction="+",
    #pol_angle=pol_angle,
    #angle_phi = np.pi/2,
    #angle_theta = np.radians(180-angle),
    #)

    

    #now define the flux monitors
    flux_monitor = td.FluxMonitor(
    center=[0, 0, central_wavelength*0.11], size=[td.inf, td.inf, 0], freqs=freqs, name="R"
    )

    flux_monitor2 = td.FluxMonitor(
    center=[0, 0, -0.11*central_wavelength +t_base], size=[td.inf, td.inf, 0], freqs=freqs, name="T"
    )


    monitor_field = td.FieldMonitor(
    center=[0,0,0],
    size= [0, np.inf,np.inf],
    freqs=freq0,
    name="field",
    )
    
    monitor_field2 = td.FieldMonitor(
    center=[d/2,0,0],
    size= [0, 0,np.inf],
    freqs=freq0,
    name="field2",
    )
    
    #point monitor for field phase
    r_phase_monitor = td.FieldMonitor(
    center=[0,0,central_wavelength*0.11],
    size= [0, 0,0],
    freqs=freq0,
    name="phase",
    )
    
    side = td.FieldMonitor(
    center=[0,0,sp+t_base/2],
    size= [0, np.inf,t_base],
    freqs=freq0,
    name="side",
    )
    top = td.FieldMonitor(
    center=[0,0,sp+h],
    size= [np.inf,np.inf,0],
    freqs=freq0,
    name="top",
    )
    bottom = td.FieldMonitor(
    center=[0,0,sp],
    size= [np.inf,np.inf,0],
    freqs=freq0,
    name="bottom",
    )
###########################################################################################################################################################################################################################
    
    
    def make_sim2(angle):
        
        
        
        #define source
        plane_wave = td.PlaneWave(
        source_time= td.GaussianPulse(freq0=freq0, fwidth=fwidth),
        size=(a, a, 0),
        center=(0, 0, 0.1*central_wavelength),
        direction="+",
        pol_angle=pol_angle,
        angle_phi = np.pi/2,
        angle_theta = np.radians(180-angle),
        )
        
        run_time = run_t / freq0  # simulation run time

        # define simulation domain size
        sim_size = (a,  a,  sim_height*central_wavelength)
        # create the Bloch boundaries
        bloch_x = td.Boundary.bloch_from_source(
        source=plane_wave, domain_size=sim_size[0], axis=0, medium=prism_material
        )
        bloch_y = td.Boundary.bloch_from_source(
        source=plane_wave, domain_size=sim_size[1], axis=1, medium=prism_material
        )

        bspec_bloch = td.BoundarySpec(x=bloch_x, y=bloch_y, z=td.Boundary.pml())
        # Calculate wavevector components and Bloch wavevectors


        grid_spec = td.GridSpec.auto(
        min_steps_per_wvl = grid_res,
        override_structures = [GoldMesh],
        wavelength=lda0,
        )
        
        
            
        if hole == 'one':
            structures=[ base, hole_1, prism, sample]
        elif hole == 'zero':
            structures = [base, prism, sample]
        elif hole == 'five':
            structures = [base,hole_1, hole_2, hole_3, hole_4, hole_5, prism, sample]
        elif hole == 'spherical':
            structures=[base,sphere,prism,sample]
        elif hole=='pores':
            
            structures = pores +[base, prism,sample]
            #structures.append(base)
            #structures.append(prism)
            #structures.append(sample)

        
        sim = td.Simulation(
        center=(0, 0, 0),
        size=sim_size,
        grid_spec=grid_spec,
        structures = structures,
        sources=[plane_wave],
        monitors=[flux_monitor, monitor_field, flux_monitor2, r_phase_monitor, side, bottom, top],
        run_time=run_time,
        boundary_spec=bspec_bloch, 
        #symmetry=(1, -1, 0),
        #shutoff=1e-7,  # reducing the default shutoff level
        )
    
        return sim

        #ax = sim.plot(z=h / 2)
        #sim.plot_grid(z=h / 2, ax=ax)

    hole_bot = sp +h
    hole_side = 0
        
        
    
    return make_sim2, hole_bot, hole_side, freq0,

def pore_scan2(
    wavelength_range,
    Nfreqs,
    source, #choose PlaneWave or Gaussian
    height,#depth of each hole
    spacing,#spacing between holes
    diameter,#hole diamater
    t_base,#metal film thickness
    metal_material,#using JC gold for now
    background_material, #hole material
    prism_material,
    sample_material,
    #angle,
    pol_angle,
    run_t,
    grid_res,
    mesh_res,
    n_prism,
    hole,
    start_angle,
    end_angle,
    steps,
    sim_height,
    length_range,
    height_range,
    thick_range,
    pores):
    
    
    make_sim, hole_bot, hole_side, freq0 = set_pore_sim2(wavelength_range,
    Nfreqs,
    source, #choose PlaneWave or Gaussian
    height,#depth of each hole
    spacing,#spacing between holes
    diameter,#hole diamater
    t_base,#metal film thickness
    metal_material,#using JC gold for now
    background_material, #hole material
    prism_material,
    sample_material,
    #angle,
    pol_angle,
    run_t,
    grid_res,
    mesh_res,
    n_prism,
    hole,
    sim_height,
    length_range,
    height_range,
    thick_range,
    pores)
    
    central_wavelength=np.mean(wavelength_range)
    angles = np.linspace(start_angle, end_angle ,steps)

        
    #plot one graph to get pricing and plot ---------------------------------------------------------------------------------------
    nano_sim = make_sim(angle=angles[0])
    print("Plotting for angle", angles[0])
    # visualize geometry
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
    nano_sim.plot(z=hole_bot, ax=ax1)
    nano_sim.plot(y=hole_side, ax=ax2)

    plt.show()
    
    task_id = web.upload(nano_sim, task_name="Nanohole Array")
    estimated_cost = web.estimate_cost(task_id=task_id)
    print("This simulation will cost", estimated_cost, "flex credits")
    print(steps, "simulations will cost", estimated_cost*steps, "flex credits")
    
    input("Proceed?")
    
    
    angles = np.linspace(start_angle, end_angle, steps)
    p_reflectivities = np.zeros(steps)
    phases = np.zeros(steps)
    phases2 = np.zeros(steps)

    sims = {f"l={l:.2f}": make_sim(l) for l in angles}
    batch = web.Batch(simulations=sims, verbose=True)

    batch_results = batch.run(path_dir="data")
    reflectivity_batch = []
    phase_batch_y = []
    phase_batch_z= []
    

    for task_name, sim_data in batch_results.items():
        trans_data = sim_data['T']
        ref_data = sim_data['R']
        phase_data = sim_data['phase']
        
        phases_y, phases_z = get_phase(phase_data)

        #calcualtes array of reflectivities at different wavelenths for each angle point
        reflectivities = reflectivity(trans_data, ref_data, freq0)

        #print(reflectivities)
            
        reflectivity_batch.append(reflectivities)
        phase_batch_y.append(phases_y)
        phase_batch_z.append(phases_z)

    #print(reflectivity_batch)
    ref = reflectivity_batch
    phasez  = phase_batch_z 
    phase = np.array(phasez)
    pz = phase
    
    #unwrap the phase for a smooth curve
    uwpz = np.degrees(np.unwrap(pz))

    #interpolate to make a continous curve
    interp_phase = CubicSpline(angles,(uwpz))

    #differentiate the curve 
    phase_div = interp_phase.derivative(1)(angles)


    #calculate the lateral goos-hanchen shift
    A = -2*np.pi / central_wavelength
    gh_shift = (1/A)*phase_div*1e-6
    
    deg_phase_z = np.degrees(phase_batch_z)
    uphase = np.unwrap(deg_phase_z)
    
    deg_phase_y = np.degrees(phase_batch_y)
    uphase_y = np.unwrap(deg_phase_y)
    
    #plt.figure()
    #plt.title("Reflectivity against angles")
    #plt.plot(angles, reflectivity_batch)
    #plt.show()
    '''
    plt.figure()
    plt.title("Point phase against angles z")
    plt.plot(angles, phase_batch_z)
    plt.show()
    
    plt.figure()
    plt.title("Lateral shift agains angles")
    plt.plot(angles, gh_shift)
    plt.show()

    ref = reflectivity_batch

    wavelength_max = np.max(wavelength_range)
    wavelength_min = np.min(wavelength_range)
    wavelengths = np.linspace(wavelength_min, wavelength_max, Nfreqs)
    
    # ref is angle x wavelength
    # ref[i][j]: i-th angle, j-th wavelength
    
    plt.figure()
    plt.plot(angles, ref)
    
    plt.xlabel("Angle (degrees)")
    plt.ylabel("Reflectivity")
    plt.title("Reflectivity vs Angle for Different Wavelengths")
    plt.grid(True)
    plt.legend()
    plt.show()
    '''

    return phase, gh_shift, ref, angles
