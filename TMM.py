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

        
        plt.figure(figsize=(8, 6))
        plt.plot(indices, ang_int_sens, marker='o')
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

    return max_gh_shift, np.argmax(gh_peaks)

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
                              
                             
                              
def effective_permittivity(rho, epsilon_m, epsilon_d):
    numerator = ((1 + rho) * epsilon_m * epsilon_d) + ((1 - rho) * epsilon_d**2)
    denominator = ((1 - rho) * epsilon_m) + ((1 + rho) * epsilon_d)
    epsilon_eff = numerator / denominator
    return epsilon_eff                                 
                              

def nk_from_perm(ereal, eim):
    emag = np.sqrt(ereal**2 + eim**2)

    n = np.sqrt((emag + ereal) / 2)
    k = np.sqrt((emag - ereal) / 2)

    return n + k*1j                          

#delete below
def scan_metal_n(
    start_angle, 
    end_angle, 
    scans, 
    n_prism,          # n1 in your existing notation
    start_metal,      # start of metal refractive index range
    end_metal,        # end of metal refractive index range
    steps, 
    n_sample,         # n3 in your existing notation
    wavelength, 
    tg, 
    ts, 
    ts_total, 
    layering, 
    graphs
):
    """
    Scans over a range of metal refractive indices (n_metal) from start_metal 
    to end_metal in 'steps' increments. At each index, it calls 'scan_angle' to 
    compute reflection phases, reflectivities, and Goos-Hänchen shifts.

    Parameters
    ----------
    start_angle : float
        Start angle (in degrees) for the angular scan.
    end_angle : float
        End angle (in degrees) for the angular scan.
    scans : int
        Number of angular points between start_angle and end_angle.
    n_prism : complex or float
        Refractive index of the prism (n1).
    start_metal : complex or float
        Starting refractive index of the metal.
    end_metal : complex or float
        Ending refractive index of the metal.
    steps : int
        Number of indices between start_metal and end_metal.
    n_sample : complex or float
        Refractive index of the sample (n3).
    wavelength : float
        Wavelength of the incident light.
    tg : float
        Thickness of the metal layer (meters).
    ts : float
        Single-layer thickness of the sample (if layering='on').
    ts_total : float
        Total thickness of the sample (if layering='on').
    layering : str
        Either 'on' or 'off', indicating whether to layer the sample in multiple layers.
    graphs : str
        Plotting mode: 'complex', 'simple', or 'shift'.
        - 'complex' plots multiple subplots (phases, GH shift, reflectivities).
        - 'simple' plots only GH shift (P) and P reflectivity.
        - 'shift' plots only the maximum differential GH shift vs. metal refractive index.

    Returns
    -------
    max_gh_shift : float
        Maximum absolute differential GH shift found among all metal indices.
    """

    # Generate the array of metal indices
    metal_indices = np.linspace(start_metal, end_metal, steps)

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

    max_diff_gh_values = []  # To store the maximum differential GH shift for each metal index
    angles = None

    # Loop through each metal refractive index
    for i, n_metal in enumerate(metal_indices):
        # Call the 'scan_angle' function with n2 replaced by n_metal
        (
            unw_pphas, 
            unw_sphas, 
            sphas, 
            pphas, 
            sref, 
            pref, 
            gh_shift_p, 
            gh_shift_s, 
            diff_gh, 
            angles
        ) = scan_angle(
            start_angle,
            end_angle,
            scans,
            n_prism,
            n_metal,
            n_sample,
            wavelength,
            tg,
            ts,
            ts_total,
            layering,
            plot='off'
        )

        # Append the results to the lists
        unw_pphas_array.append(unw_pphas)
        unw_sphas_array.append(unw_sphas)
        sphas_array.append(sphas)
        pphas_array.append(pphas)
        sref_array.append(sref)
        pref_array.append(pref)
        gh_shift_p_array.append(gh_shift_p)
        gh_shift_s_array.append(gh_shift_s)
        diff_gh_array.append(diff_gh)

        # Label for plotting
        labels.append(f'n_metal = {n_metal:.3f}')

        # Compute maximum absolute differential GH shift for this metal index
        max_diff_gh = np.max(np.abs(diff_gh))
        max_diff_gh_values.append(max_diff_gh)

    # ------------------
    #       PLOTTING
    # ------------------
    if graphs == 'complex':
        # After the loop, plot the desired values for each refractive index
        num_plots = 9  # S phase (wrapped/unwrapped), P phase (wrapped/unwrapped),
                       # GH shifts (S, P, differential), S reflectivity, P reflectivity
        num_cols = 3
        num_rows = int(np.ceil(num_plots / num_cols))

        # Create subplots
        fig, axs = plt.subplots(num_rows, num_cols, figsize=(18, num_rows * 4))
        axs = axs.flatten()  # Flatten the array of axes for easy indexing

        # Plot configurations (including unwrapped phases)
        plot_data = [
            (sphas_array,      'Wrapped S Phase vs Angle', 'Wrapped S Phase'),
            (pphas_array,      'Wrapped P Phase vs Angle', 'Wrapped P Phase'),
            (unw_sphas_array,  'Unwrapped S Phase vs Angle', 'Unwrapped S Phase'),
            (unw_pphas_array,  'Unwrapped P Phase vs Angle', 'Unwrapped P Phase'),
            (gh_shift_p_array, 'GH Shift (P) vs Angle', 'GH Shift P'),
            (gh_shift_s_array, 'GH Shift (S) vs Angle', 'GH Shift S'),
            (diff_gh_array,    'Differential GH Shift vs Angle', 'GH Shift (P-S)'),
            (sref_array,       'S Reflectivity vs Angle', 'Reflectivity'),
            (pref_array,       'P Reflectivity vs Angle', 'Reflectivity'),
        ]

        # Loop over each subplot
        for ax_idx, (data_array, title, ylabel) in enumerate(plot_data):
            for data, label in zip(data_array, labels):
                axs[ax_idx].plot(angles, data, label=label)
            axs[ax_idx].set_title(title)
            axs[ax_idx].set_xlabel('Angle (degrees)')
            axs[ax_idx].set_ylabel(ylabel)
            axs[ax_idx].grid(True)
            axs[ax_idx].legend(loc='upper left', bbox_to_anchor=(1, 1))

        # Hide any unused subplots
        for idx in range(len(plot_data), len(axs)):
            fig.delaxes(axs[idx])

        plt.tight_layout()
        plt.show()

    elif graphs == 'simple':
        # Only plot GH shift (P) and P reflectivity
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        # GH shift (P)
        for data, label in zip(gh_shift_p_array, labels):
            axs[0].plot(angles, data, label=label)
        axs[0].set_title('GH Shift (P) vs Angle')
        axs[0].set_xlabel('Angle (degrees)')
        axs[0].set_ylabel('GH Shift P')
        axs[0].grid(True)
        axs[0].legend(loc='upper left', bbox_to_anchor=(1, 1))

        # P Reflectivity
        for data, label in zip(pref_array, labels):
            axs[1].plot(angles, data, label=label)
        axs[1].set_title('P Reflectivity vs Angle')
        axs[1].set_xlabel('Angle (degrees)')
        axs[1].set_ylabel('P Reflectivity')
        axs[1].grid(True)
        axs[1].legend(loc='upper left', bbox_to_anchor=(1, 1))

        plt.tight_layout()
        plt.show()

    elif graphs == 'shift':
        # Plot maximum differential GH shift vs metal refractive index
        plt.figure(figsize=(8, 6))
        plt.plot(metal_indices, max_diff_gh_values, marker='o')
        plt.title('Maximum Differential GH Shift vs Metal Refractive Index')
        plt.xlabel('Metal Refractive Index')
        plt.ylabel('Maximum Differential GH Shift')
        plt.grid(True)
        plt.legend(['Max Diff GH Shift'], loc='upper left', bbox_to_anchor=(1, 1))
        plt.tight_layout()
        plt.show()

    # ------------------
    # RETURN GH SHIFT(S)
    # ------------------
    gh_sens = []
    gh_peaks = []

    # For each metal index array, compute GH at resonance and maximum absolute GH shift
    for i in range(len(metal_indices)):
        # The resonance is considered where reflectivity (pref_array[i]) is minimum
        res_ind = np.argmin(pref_array[i])
        gh_at_spr = diff_gh_array[i][res_ind]
        gh_sens.append(gh_at_spr)

        # The maximum absolute GH shift across all angles for this n_metal
        gh_peak_ind = np.argmax(np.abs(diff_gh_array[i]))
        gh_peak_val = diff_gh_array[i][gh_peak_ind]
        gh_peaks.append(gh_peak_val)

    # Return the maximum absolute GH shift (largest positive or negative) among all scans
    max_gh_shift = np.max(np.abs(gh_peaks))

    return max_gh_shift
                          
def scan_metal_n_list(
    metal_n_list, 
    start_angle, 
    end_angle, 
    scans, 
    n_prism,         # n1 in your existing notation
    n_sample,        # n3 in your existing notation
    wavelength, 
    tg, 
    ts, 
    ts_total, 
    layering, 
    graphs
):
    """
    Scans over a custom list/array of metal refractive indices (complex or real).
    For each metal refractive index in 'metal_n_list', calls 'scan_angle' and
    accumulates phase, reflectivity, and Goos-Hänchen shift results.

    Parameters
    ----------
    metal_n_list : list or np.ndarray
        A list/array of metal refractive indices (complex). Example:
        [(0.18344+3.4332j), (0.1592+2.6516j), (1.33+0j), ...]
    start_angle : float
        Start angle (in degrees) for the angular scan.
    end_angle : float
        End angle (in degrees) for the angular scan.
    scans : int
        Number of angular points between start_angle and end_angle.
    n_prism : float or complex
        Refractive index of the prism (n1).
    n_sample : float or complex
        Refractive index of the sample (n3).
    wavelength : float
        Wavelength of the incident light (in the same length units as thickness).
    tg : float
        Thickness of the metal layer (meters).
    ts : float
        Single-layer thickness of the sample (if layering='on').
    ts_total : float
        Total thickness of the sample (if layering='on').
    layering : str
        Either 'on' or 'off', indicating whether to layer the sample in multiple layers.
    graphs : str
        Plotting mode: 'complex', 'simple', 'shift', or anything else to disable plotting.
        - 'complex': multiple subplots of phases, GH shifts, and reflectivities.
        - 'simple': only plots GH shift (P) and P reflectivity.
        - 'shift': only plots the maximum differential GH shift vs. metal index index (in the list).

    Returns
    -------
    max_gh_shift : float
        The maximum absolute differential GH shift found among all metal n values.
    """

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

    # To store the maximum differential GH shift for each metal index
    max_diff_gh_values = []
    angles = None

    # Loop through each metal refractive index in the list
    for i, n_metal in enumerate(metal_n_list):
        # Call the existing 'scan_angle' function 
        (
            unw_pphas, 
            unw_sphas, 
            sphas, 
            pphas, 
            sref, 
            pref, 
            gh_shift_p, 
            gh_shift_s, 
            diff_gh, 
            angles
        ) = scan_angle(
            start_angle=start_angle,
            end_angle=end_angle,
            scans=scans,
            n1=n_prism,       # Prism index
            n2=n_metal,       # Metal index (variable here)
            n3=n_sample,      # Sample index
            wavelength=wavelength,
            tg=tg,
            ts=ts,
            ts_total=ts_total,
            layering=layering,
            plot='off'
        )

        # Collect the results
        unw_pphas_array.append(unw_pphas)
        unw_sphas_array.append(unw_sphas)
        sphas_array.append(sphas)
        pphas_array.append(pphas)
        sref_array.append(sref)
        pref_array.append(pref)
        gh_shift_p_array.append(gh_shift_p)
        gh_shift_s_array.append(gh_shift_s)
        diff_gh_array.append(diff_gh)

        # Make a label for plotting
        labels.append(f"n_metal = {n_metal}")

        # Compute the maximum absolute differential GH shift for this specific metal index
        max_diff_gh = np.max(np.abs(diff_gh))
        max_diff_gh_values.append(max_diff_gh)

    # ------------------
    #       PLOTTING
    # ------------------
    if graphs == 'complex':
        # 9 total plots: wrapped S/P phases, unwrapped S/P phases, GH shifts (S, P, diff), S & P reflectivities
        num_plots = 9
        num_cols = 3
        num_rows = int(np.ceil(num_plots / num_cols))

        fig, axs = plt.subplots(num_rows, num_cols, figsize=(18, num_rows * 4))
        axs = axs.flatten()

        plot_data = [
            (sphas_array,      'Wrapped S Phase vs Angle', 'Wrapped S Phase'),
            (pphas_array,      'Wrapped P Phase vs Angle', 'Wrapped P Phase'),
            (unw_sphas_array,  'Unwrapped S Phase vs Angle', 'Unwrapped S Phase'),
            (unw_pphas_array,  'Unwrapped P Phase vs Angle', 'Unwrapped P Phase'),
            (gh_shift_p_array, 'GH Shift (P) vs Angle', 'GH Shift P'),
            (gh_shift_s_array, 'GH Shift (S) vs Angle', 'GH Shift S'),
            (diff_gh_array,    'Differential GH Shift vs Angle', 'GH Shift (P-S)'),
            (sref_array,       'S Reflectivity vs Angle', 'Reflectivity'),
            (pref_array,       'P Reflectivity vs Angle', 'Reflectivity'),
        ]

        for ax_idx, (data_array, title, ylabel) in enumerate(plot_data):
            for data, label in zip(data_array, labels):
                axs[ax_idx].plot(angles, data, label=label)
            axs[ax_idx].set_title(title)
            axs[ax_idx].set_xlabel('Angle (degrees)')
            axs[ax_idx].set_ylabel(ylabel)
            axs[ax_idx].grid(True)
            axs[ax_idx].legend(loc='upper left', bbox_to_anchor=(1, 1))

        # Hide any unused subplots (in case num_plots < len(axs))
        for idx in range(len(plot_data), len(axs)):
            fig.delaxes(axs[idx])

        plt.tight_layout()
        plt.show()

    elif graphs == 'simple':
        # Only plot GH shift (P) and P reflectivity
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        # GH Shift (P)
        for data, label in zip(gh_shift_p_array, labels):
            axs[0].plot(angles, data, label=label)
        axs[0].set_title('GH Shift (P) vs Angle')
        axs[0].set_xlabel('Angle (degrees)')
        axs[0].set_ylabel('GH Shift P')
        axs[0].grid(True)
        axs[0].legend(loc='upper left', bbox_to_anchor=(1, 1))

        # P Reflectivity
        for data, label in zip(pref_array, labels):
            axs[1].plot(angles, data, label=label)
        axs[1].set_title('P Reflectivity vs Angle')
        axs[1].set_xlabel('Angle (degrees)')
        axs[1].set_ylabel('P Reflectivity')
        axs[1].grid(True)
        axs[1].legend(loc='upper left', bbox_to_anchor=(1, 1))

        plt.tight_layout()
        plt.show()

    elif graphs == 'shift':
        # Plot maximum differential GH shift vs index *position* in the list
        # (We could also parse real/imag parts, but that might be less straightforward.)
        plt.figure(figsize=(8, 6))
        index_positions = np.arange(len(metal_n_list))  # 0, 1, 2, ...
        plt.plot(index_positions, max_diff_gh_values, marker='o')
        plt.title('Maximum Differential GH Shift vs Metal Index (List Position)')
        plt.xlabel('Index in metal_n_list')
        plt.ylabel('Maximum Differential GH Shift')
        plt.grid(True)
        plt.legend(['Max Diff GH Shift'], loc='upper left', bbox_to_anchor=(1, 1))
        plt.tight_layout()
        plt.show()

    # ---------------
    #  RETURN VALUE
    # ---------------
    # Also compute GH shift at resonance and track the maximum absolute GH shift
    gh_sens = []
    gh_peaks = []

    for i in range(len(metal_n_list)):
        # Resonance index is where P-reflectivity is minimum
        res_ind = np.argmin(pref_array[i])
        gh_at_spr = diff_gh_array[i][res_ind]
        gh_sens.append(gh_at_spr)

        # The maximum absolute GH shift across all angles for this particular n_metal
        gh_peak_ind = np.argmax(np.abs(diff_gh_array[i]))
        gh_peak_val = diff_gh_array[i][gh_peak_ind]
        gh_peaks.append(gh_peak_val)

    # The global maximum
    max_gh_shift = np.max(np.abs(gh_peaks))

    return max_gh_shift, np.argmax(max_diff_gh_values)


def scan_anglex(start_angle, end_angle, scans, n1, n2, n3, wavelength, tg, ts, ts_total, layering, plot):
    angles = np.linspace(start_angle, end_angle, scans)

    sphas = []
    pphas = []
    sref = []
    pref = []

    for angle in angles:
        # Convert degrees to radians
        incident_angle = np.radians(angle)

        # Call your TMM_setup function here
        spha, ppha, sre, pre = TMM_setup(n1, n2, n3, incident_angle, wavelength, tg, ts, ts_total, layering)
        sphas.append(spha)
        pphas.append(ppha)
        sref.append(sre)
        pref.append(pre)

    sphas = np.array(sphas)
    pphas = np.array(pphas)
    sref = np.array(sref)
    pref = np.array(pref)

    # Unwrap the phase in degrees
    unw_pphas = np.degrees(np.unwrap(pphas))
    unw_sphas = np.degrees(np.unwrap(sphas))

    # Calculate Goos-Hänchen shifts
    gh_shift_p = ((-1 * wavelength) / (2 * np.pi)) * np.gradient(unw_pphas, angles)
    gh_shift_s = ((-1 * wavelength) / (2 * np.pi)) * np.gradient(unw_sphas, angles)
    diff_gh = gh_shift_p - gh_shift_s

    # Full 3x3 plot (existing plotting mode)
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
        axes[2, 1].plot(angles, sref, label='S reflectivity')
        axes[2, 1].legend()
        axes[2, 1].set_title("Reflectivity")

        # S reflectivity
        axes[2, 2].plot(angles, sref)
        axes[2, 2].set_title("S reflectivity")

        plt.tight_layout()
        plt.show()
        print(np.min(diff_gh))

    # Simple 1x2 plot with reflectivity and differential GH shift
    if plot == 'simple':
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        
        # Reflectivity (both P and S)
        axes[0].plot(angles, pref, label='P reflectivity')
        axes[0].plot(angles, sref, label='S reflectivity')
        axes[0].set_xlabel("Angle (degrees)")
        axes[0].set_ylabel("Reflectivity")
        axes[0].set_title("Reflectivity")
        axes[0].legend()

        # Differential GH shift
        axes[1].plot(angles, diff_gh, label='GH shift (P-S)')
        axes[1].set_xlabel("Angle (degrees)")
        axes[1].set_ylabel("GH shift (metres)")
        axes[1].set_title("Differential Goos-Hänchen shift")
        axes[1].legend()

        plt.tight_layout()
        plt.show()

    return unw_pphas, unw_sphas, sphas, pphas, sref, pref, gh_shift_p, gh_shift_s, diff_gh, angles

#this function scans all n, k and metal layer thickness values in a 3D paramater sweep that maximise the lateral shift
def metal_opt(
    metal_n_list,
    start_angle, 
    end_angle, 
    scans, 
    n_prism,         # n1 in your existing notation
    n_sample,        # n3 in your existing notation
    wavelength, 
    tg, 
    ts, 
    ts_total, 
    layering, 
    graphs
):
    """
    Scans over a custom list/array of metal refractive indices (complex or real).
    For each metal refractive index in 'metal_n_list', calls 'scan_angle' and
    accumulates phase, reflectivity, and Goos-Hänchen shift results.

    Parameters
    ----------
    metal_n_list : list or np.ndarray
        A list/array of metal refractive indices (complex). Example:
        [(0.18344+3.4332j), (0.1592+2.6516j), (1.33+0j), ...]
    start_angle : float
        Start angle (in degrees) for the angular scan.
    end_angle : float
        End angle (in degrees) for the angular scan.
    scans : int
        Number of angular points between start_angle and end_angle.
    n_prism : float or complex
        Refractive index of the prism (n1).
    n_sample : float or complex
        Refractive index of the sample (n3).
    wavelength : float
        Wavelength of the incident light (in the same length units as thickness).
    tg : float
        Thickness of the metal layer (meters).
    ts : float
        Single-layer thickness of the sample (if layering='on').
    ts_total : float
        Total thickness of the sample (if layering='on').
    layering : str
        Either 'on' or 'off', indicating whether to layer the sample in multiple layers.
    graphs : str
        Plotting mode: 'complex', 'simple', 'shift', or anything else to disable plotting.
        - 'complex': multiple subplots of phases, GH shifts, and reflectivities.
        - 'simple': only plots GH shift (P) and P reflectivity.
        - 'shift': only plots the maximum differential GH shift vs. metal index index (in the list).

    Returns
    -------
    max_gh_shift : float
        The maximum absolute differential GH shift found among all metal n values.
    """

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

    # To store the maximum differential GH shift for each metal index
    max_diff_gh_values = []
    angles = None

    
    # Loop through each metal refractive index in the list
    for i, n_metal in enumerate(metal_n_list):
        # Call the existing 'scan_angle' function 
        (
            unw_pphas, 
            unw_sphas, 
            sphas, 
            pphas, 
            sref, 
            pref, 
            gh_shift_p, 
            gh_shift_s, 
            diff_gh, 
            angles
        ) = scan_angle(
            start_angle=start_angle,
            end_angle=end_angle,
            scans=scans,
            n1=n_prism,       # Prism index
            n2=n_metal,       # Metal index (variable here)
            n3=n_sample,      # Sample index
            wavelength=wavelength,
            tg=tg,
            ts=ts,
            ts_total=ts_total,
            layering=layering,
            plot='off'
        )

        # Collect the results
        unw_pphas_array.append(unw_pphas)
        unw_sphas_array.append(unw_sphas)
        sphas_array.append(sphas)
        pphas_array.append(pphas)
        sref_array.append(sref)
        pref_array.append(pref)
        gh_shift_p_array.append(gh_shift_p)
        gh_shift_s_array.append(gh_shift_s)
        diff_gh_array.append(diff_gh)

        # Make a label for plotting
        labels.append(f"n_metal = {n_metal}")

        # Compute the maximum absolute differential GH shift for this specific metal index
        max_diff_gh = np.max(np.abs(diff_gh))
        max_diff_gh_values.append(max_diff_gh)

    # ------------------
    #       PLOTTING
    # ------------------
    if graphs == 'complex':
        # 9 total plots: wrapped S/P phases, unwrapped S/P phases, GH shifts (S, P, diff), S & P reflectivities
        num_plots = 9
        num_cols = 3
        num_rows = int(np.ceil(num_plots / num_cols))

        fig, axs = plt.subplots(num_rows, num_cols, figsize=(18, num_rows * 4))
        axs = axs.flatten()

        plot_data = [
            (sphas_array,      'Wrapped S Phase vs Angle', 'Wrapped S Phase'),
            (pphas_array,      'Wrapped P Phase vs Angle', 'Wrapped P Phase'),
            (unw_sphas_array,  'Unwrapped S Phase vs Angle', 'Unwrapped S Phase'),
            (unw_pphas_array,  'Unwrapped P Phase vs Angle', 'Unwrapped P Phase'),
            (gh_shift_p_array, 'GH Shift (P) vs Angle', 'GH Shift P'),
            (gh_shift_s_array, 'GH Shift (S) vs Angle', 'GH Shift S'),
            (diff_gh_array,    'Differential GH Shift vs Angle', 'GH Shift (P-S)'),
            (sref_array,       'S Reflectivity vs Angle', 'Reflectivity'),
            (pref_array,       'P Reflectivity vs Angle', 'Reflectivity'),
        ]

        for ax_idx, (data_array, title, ylabel) in enumerate(plot_data):
            for data, label in zip(data_array, labels):
                axs[ax_idx].plot(angles, data, label=label)
            axs[ax_idx].set_title(title)
            axs[ax_idx].set_xlabel('Angle (degrees)')
            axs[ax_idx].set_ylabel(ylabel)
            axs[ax_idx].grid(True)
            axs[ax_idx].legend(loc='upper left', bbox_to_anchor=(1, 1))

        # Hide any unused subplots (in case num_plots < len(axs))
        for idx in range(len(plot_data), len(axs)):
            fig.delaxes(axs[idx])

        plt.tight_layout()
        plt.show()

    elif graphs == 'simple':
        # Only plot GH shift (P) and P reflectivity
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        # GH Shift (P)
        for data, label in zip(gh_shift_p_array, labels):
            axs[0].plot(angles, data, label=label)
        axs[0].set_title('GH Shift (P) vs Angle')
        axs[0].set_xlabel('Angle (degrees)')
        axs[0].set_ylabel('GH Shift P')
        axs[0].grid(True)
        axs[0].legend(loc='upper left', bbox_to_anchor=(1, 1))

        # P Reflectivity
        for data, label in zip(pref_array, labels):
            axs[1].plot(angles, data, label=label)
        axs[1].set_title('P Reflectivity vs Angle')
        axs[1].set_xlabel('Angle (degrees)')
        axs[1].set_ylabel('P Reflectivity')
        axs[1].grid(True)
        axs[1].legend(loc='upper left', bbox_to_anchor=(1, 1))

        plt.tight_layout()
        plt.show()

    elif graphs == 'shift':
        # Plot maximum differential GH shift vs index *position* in the list
        # (We could also parse real/imag parts, but that might be less straightforward.)
        plt.figure(figsize=(8, 6))
        index_positions = np.arange(len(metal_n_list))  # 0, 1, 2, ...
        plt.plot(index_positions, max_diff_gh_values, marker='o')
        plt.title('Maximum Differential GH Shift vs Metal Index (List Position)')
        plt.xlabel('Index in metal_n_list')
        plt.ylabel('Maximum Differential GH Shift')
        plt.grid(True)
        plt.legend(['Max Diff GH Shift'], loc='upper left', bbox_to_anchor=(1, 1))
        plt.tight_layout()
        plt.show()

    # ---------------
    #  RETURN VALUE
    # ---------------
    # Also compute GH shift at resonance and track the maximum absolute GH shift
    gh_sens = []
    gh_peaks = []

    for i in range(len(metal_n_list)):
        # Resonance index is where P-reflectivity is minimum
        res_ind = np.argmin(pref_array[i])
        gh_at_spr = diff_gh_array[i][res_ind]
        gh_sens.append(gh_at_spr)

        # The maximum absolute GH shift across all angles for this particular n_metal
        gh_peak_ind = np.argmax(np.abs(diff_gh_array[i]))
        gh_peak_val = diff_gh_array[i][gh_peak_ind]
        gh_peaks.append(gh_peak_val)

    # The global maximum
    max_gh_shift = np.max(np.abs(gh_peaks))

    return  index_position, max_diff_gh_values


def metal opt2(
    metal_n_list,
    metal_thick_list,
    start_angle, 
    end_angle, 
    scans, 
    n_prism,         # n1 in your existing notation
    n_sample,        # n3 in your existing notation
    wavelength, 
    tg, 
    ts, 
    ts_total, 
    layering, 
    graphs
):

for i in range(len(metal_thick_list)):

    index_position, gh_val = metal opt(
    metal_n_list,
    start_angle, 
    end_angle, 
    scans, 
    n_prism,         # n1 in your existing notation
    n_sample,        # n3 in your existing notation
    wavelength, 
    tg, 
    ts, 
    metal_thick_list[i], 
    layering, 
    graphs)
    

