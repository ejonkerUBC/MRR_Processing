import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline
import os

import processing_utils as utils # in the process of moving funcitons over here

def find_csv_for_ring(base_path, ring_num):
    if CONFIG["Tapeout"] == 1:
        target_str = f'Ring{ring_num}'
    elif CONFIG["Tapeout"] == 2:
        target_str = f'Ring{ring_num}rad'
    elif CONFIG["Tapeout"] == 3:
        # target_str = f'Ring{ring_num}rad'
        target_str = f'Racetrack{ring_num}'
    print(f"Targetting {target_str}")
    for folder_name in os.listdir(base_path):
        if target_str in folder_name:
            folder_path = os.path.join(base_path, folder_name)
            if os.path.isdir(folder_path):
                csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
                if csv_files:
                    return os.path.join(folder_path, csv_files[0]), folder_name
                else:
                    print(f"No CSV file found in folder: {folder_path}")
                    return None, folder_name
    print(f"No folder containing '{target_str}' found in {base_path}")
    return None, None

def load_data_from_specific_lines(file_path, dvc):
    c = 299792458  # speed of light (m/s)

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Extract wavelength line
    wavelength_line = lines[23].strip().split(',')[1:]

    # Select the correct spectrum line depending on tapeout and device type
    if CONFIG["Tapeout"] == 1:
        spectrum_line = lines[26].strip().split(',')[1:] if dvc == 'ring' else lines[25].strip().split(',')[1:]
    elif CONFIG["Tapeout"] in [2, 3]:
        spectrum_line = lines[24].strip().split(',')[1:] if dvc == 'ring' else lines[25].strip().split(',')[1:]

    # Convert to numpy arrays
    wavelength = np.array(wavelength_line, dtype=float)  # in nm
    spectrum = np.array(spectrum_line, dtype=float)

    # Convert wavelength to frequency (THz)
    wavelength_m = wavelength * 1e-9  # nm → m
    frequency = (c / wavelength_m) * 1e-12  # Hz → THz

    return wavelength, frequency, spectrum

def calculate_r_squared(y_obs, y_fit):
    ss_res = np.sum((y_obs - y_fit) ** 2)
    ss_tot = np.sum((y_obs - np.mean(y_obs)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    return r_squared

def estimate_background_poly(wavelength_window, spectrum_window, degree=3, num_edge_points=5):
    """
    Fit a polynomial to the edge points of the window to estimate background.
    This avoids including the resonance dip in the background fit.
    """
    if 2 * num_edge_points >= len(wavelength_window):
        raise ValueError("Not enough points to estimate background from edges.")

    x_edges = np.concatenate([wavelength_window[:num_edge_points], wavelength_window[-num_edge_points:]])
    y_edges = np.concatenate([spectrum_window[:num_edge_points], spectrum_window[-num_edge_points:]])
    
    coeffs = np.polyfit(x_edges, y_edges, degree)
    poly_background = np.polyval(coeffs, wavelength_window)
    
    return poly_background, coeffs

def Calculate_roots(finesse_array,ER_lin_array):
    # Calculate A, B, root1, root2 ({root1,root2} = {alpha,t}, but are indestinguishable from the calculation)
    A_values = np.cos(np.pi / finesse_array) / (1 + np.sin(np.pi / finesse_array))
    B_values = 1 - (1 / ER_lin_array) * ((1 - np.cos(np.pi / finesse_array)) / (1 + np.cos(np.pi / finesse_array)))
    root1_values = np.sqrt(A_values / B_values) + np.sqrt((A_values / B_values) - A_values)
    root2_values = np.sqrt(A_values / B_values) - np.sqrt((A_values / B_values) - A_values)
    sigma1_values = np.sqrt(1-root1_values**2)
    sigma2_values = np.sqrt(1-root2_values**2)
    return root1_values, root2_values, sigma1_values, sigma2_values

def Plot_and_Save_Spectra(ring_wl,ring_frq,ring_spec,figures_dir):
    
    if CONFIG["Subtract_GC"]:
        GC_spec = load_GC_spectrum()
        corrected_spec = ring_spec - GC_spec
    else:
        corrected_spec = ring_spec
    
    # Find initial resonance dips
    min_resonance_spacing_nm = 0.3  # Minimum expected spacing between resonances (adjust as needed)
    nm_per_point = np.mean(np.diff(ring_wl))
    min_distance_pts = int(min_resonance_spacing_nm / nm_per_point)
    peaks, _ = find_peaks(-corrected_spec, prominence=CONFIG["res_ER_thrshld"], distance=min_distance_pts)
    in_range_peaks = [p for p in peaks if CONFIG["lower_wl_limit"] <= ring_wl[p] <= CONFIG["upper_wl_limit"]]
    out_of_range_peaks = [p for p in peaks if p not in in_range_peaks]

    # === Spline Baseline Correction Using Midpoints Between Resonances ===
    try:
        if len(in_range_peaks) >= 2:
            mid_indices = []
            ptgap = peaks[1]-peaks[0]
            strt = ((peaks[0] + peaks[1]) // 2)-1
            for k in range(int(np.floor(strt/ptgap))):
                mid_indices.append(strt-ptgap*(k+1))
            for i in range(len(peaks) - 1):
                left = peaks[i]
                right = peaks[i + 1]
                midpoint = (left + right) // 2
                mid_indices.append(midpoint)
            ptgap = peaks[-1]-peaks[-2]
            strt = (peaks[-2] + peaks[-1]) // 2
            for k in range(int(np.floor(((len(ring_spec)-strt)-1)/ptgap))):
                mid_indices.append(strt+ptgap*(k+1))
            mid_indices.sort()

            if mid_indices:
                spline_fit = UnivariateSpline(ring_wl[mid_indices], corrected_spec[mid_indices], s=0.01) #<<<< change this depending on how much the spline should change
                baseline = spline_fit(ring_wl)
                
                flattened_spec = corrected_spec - baseline
            else:
                flattened_spec = corrected_spec
                print("no mid_indicies")
        else:
            flattened_spec = corrected_spec  # Fallback for fewer than 2 resonances
            print("less than 2 peaks")
    except Exception as e:
        print(f"Baseline correction failed: {e}")
        flattened_spec = corrected_spec

    # Re-find peaks on flattened spectrum
    if not(CONFIG["Flatten_Spectrum"]):
        flattened_spec = corrected_spec
        
    min_resonance_spacing_nm = 0.3  # Minimum expected spacing between resonances (adjust as needed)
    nm_per_point = np.mean(np.diff(ring_wl))
    min_distance_pts = int(min_resonance_spacing_nm / nm_per_point)
    peaks, _ = find_peaks(-flattened_spec, prominence=CONFIG["res_ER_thrshld"], distance=min_distance_pts)
    in_range_peaks = [p for p in peaks if CONFIG["lower_wl_limit"] <= ring_wl[p] <= CONFIG["upper_wl_limit"]]
    out_of_range_peaks = [p for p in peaks if p not in in_range_peaks]
    
    # print(mid_indices)
    
    plt.figure()
    plt.plot(ring_wl,ring_spec,color='navy')
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Power (dBm)")
    plt.title("MRR Spectrum")
    # plt.plot(ring_wl[mid_indices],baseline[mid_indices],'-o',markerfacecolor=None)
    
    plt.figure()
    C26 = 192.6
    C28 = 192.8
    C30 = 193.0
    C32 = 193.2
    C34 = 193.4
    C36 = 193.6
    C38 = 193.8
    C40 = 194.0
    C42 = 194.2
    
    wdm = [C26,C28,C30,C32,C34,C36,C38,C40,C42]
    
    plt.plot(ring_frq,ring_spec,color='black')
    ax = plt.gca()
    
    choice = 1
    # plt.axvline([C34],linestyle=':',color='red')
    # plt.axvline([C34+0.075],linestyle='--',color='red')
    # plt.axvline([C34-0.075],linestyle='--',color='red')
    if choice ==1:
        rect = plt.Rectangle((C34-0.075, -45), 0.15, 25,facecolor='red', edgecolor='red', alpha=0.5)
        ax.add_patch(rect)
        plt.title("Notch Filter",color='red')
    
    # plt.axvline([C34],linestyle=':',color='blue')
    # plt.axvline([C34+0.100],linestyle='--',color='blue')
    # plt.axvline([C34-0.100],linestyle='--',color='blue')
    if choice==2:
        rect = plt.Rectangle((C34-0.1, -45), 0.2, 25,facecolor='blue', edgecolor='blue',alpha=0.5)
        ax.add_patch(rect)
        rect = plt.Rectangle((C34+0.1, -45), 1.2, 25,facecolor='cyan', edgecolor='cyan',alpha=0.5)
        ax.add_patch(rect)
        rect = plt.Rectangle((C34+0.1-2.8, -45), 2.6, 25,facecolor='cyan', edgecolor='cyan',alpha=0.5)
        ax.add_patch(rect)
        plt.title("BP Filter",color='blue')
        
    if choice == 3:
        for lw in wdm:
            rect = plt.Rectangle((lw-0.1, -45), 0.2, 25,facecolor='lightgreen', edgecolor='green')
            ax.add_patch(rect)
            plt.title("WDM Filter",color='green')
            
            # plt.axvline([lw],linestyle=':',color='green')
            # plt.axvline([lw+0.100],linestyle='--',color='green')
            # plt.axvline([lw-0.100],linestyle='--',color='green')

    plt.xlabel("Frequency (THz)")
    plt.ylabel("Power (dBm)")
    # plt.title("MRR Spectrum")
    
    # Plot spectrum with resonances
    plt.figure(figsize=(6, 5))
    plt.plot(ring_wl, flattened_spec, label='Spectrum')
    plt.plot(ring_wl[in_range_peaks], flattened_spec[in_range_peaks], 'ro', label='In-Range Resonances')
    plt.plot(ring_wl[out_of_range_peaks], flattened_spec[out_of_range_peaks], 'ro', mfc='none')
    plt.axvspan(ring_wl[0], CONFIG["lower_wl_limit"], color='gray', alpha=0.2)
    plt.axvspan(CONFIG["upper_wl_limit"], ring_wl[-1], color='gray', alpha=0.2)
    plt.axvline(CONFIG["lower_wl_limit"], color='dimgray', linestyle='--')
    plt.axvline(CONFIG["upper_wl_limit"], color='dimgray', linestyle='--')
    plt.xlim([ring_wl[0], ring_wl[-1]])
    plt.xlabel('Wavelength (nm)');   plt.ylabel('Power (dBm)')
    plt.title('Flattened Ring Resonator Spectrum') if CONFIG["Flatten_Spectrum"] else plt.title('Ring Resonator Spectrum')
    plt.legend(); 
    plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "flattened_spectrum_with_resonances.png"))
    plt.show() if CONFIG["Show_Other_Plots"] else plt.close()
    return in_range_peaks, flattened_spec

def Plot_and_Save_Resonance(in_range_peaks, x_data, y_data_dBm, x_smooth, fit_y_dBm_smooth, ResSpcs, figures_dir):
    # Plot each resonance
    print(f"Resonance {ResSpcs[-1]["peak_index"]+1}/{len(in_range_peaks)}:")
    plt.figure(figsize=(8, 4))
    plt.plot(x_data-ResSpcs[-1]["wavelength"], y_data_dBm, 'b.', label='Data')
    plt.plot(x_smooth-ResSpcs[-1]["wavelength"], fit_y_dBm_smooth, 'r-', label='Lorentzian Fit')
    plt.axvline(0, color='g', linestyle='--', label=f'Resonance = {ResSpcs[-1]["wavelength"]:.2f} nm')
    plt.xlabel('Wavelength detuning (nm)'); plt.ylabel('Power (dBm)'); plt.ticklabel_format(useOffset=False)
    plt.title(f'Resonance {ResSpcs[-1]["peak_index"]+1}\nFWHM = {ResSpcs[-1]["FWHM"]:.4f}+-{ResSpcs[-1]["fwhm_err"]:.4f} nm,   Q = {ResSpcs[-1]["Q"]:.2f}+-{ResSpcs[-1]["Q_err"]:.2f},   ER = {ResSpcs[-1]["ER_dB"]:.3f} dBm')
    plt.legend(); plt.grid(True); plt.tight_layout()

    # Save and show figure
    filename = f"resonance_{ResSpcs[-1]["peak_index"]+1}_fit_{ResSpcs[-1]["wavelength"]:.2f}nm.png"
    plt.savefig(os.path.join(figures_dir, filename))
    plt.show() if CONFIG["Show_Resonance_Plots"] else plt.close()

def load_GC_spectrum():
    GC_1525_path = r'C:\Users\Evan\Documents\ECE Masters\BTO Project\ShEtch Tapeout 2\sweepLaser\finesweepLaser\GCloopbackCorner1Wvl1550\30-May-2025 11.56.47.csv'
    GC_1550_path = r'C:\Users\Evan\Documents\ECE Masters\BTO Project\ShEtch Tapeout 2\sweepLaser\finesweepLaser\GCloopbackCorner1Wvl1525\30-May-2025 11.55.34.csv'
    GC_1575_path = r'C:\Users\Evan\Documents\ECE Masters\BTO Project\ShEtch Tapeout 2\sweepLaser\finesweepLaser\GCloopbackCorner1Wvl1575\30-May-2025 11.57.27.csv'
    
    if CONFIG["ring_number"] in range(0,20):
        GC_csv_path = GC_1525_path
    elif CONFIG["ring_number"] in range(20,40):
        GC_csv_path = GC_1550_path
    elif CONFIG["ring_number"] in range(40,60):
        GC_csv_path = GC_1575_path
    GC_wl, GC_frq, GC_spec = load_data_from_specific_lines(GC_csv_path, dvc='GC')
    return GC_spec

def style_plot(ax, title, xlabel, ylabel):
    ax.set_title(title) if title is not None else None
    ax.set_xlabel(xlabel) if xlabel is not None else None
    ax.set_ylabel(ylabel) if ylabel is not None else None
    ax.grid(True)
    return

def configure():
    if CONFIG["Tapeout"] == 1:
        '''
        Configuration for BTO tapeout 1 data
        '''
        csv_path, folder_name = find_csv_for_ring(base_path + r"\sweepLaser run 1", CONFIG["ring_number"])
        if csv_path:
            print(f"CSV file path: {csv_path}")    
        figures_dir = os.path.join(base_path, "RingResults_coarse", folder_name)
        Ring_Rad = 50
        print(f"Ring Radius = {Ring_Rad}")
        ring_radius_m = Ring_Rad * 1e-6  # um to meters
        ring_length = 2 * np.pi * ring_radius_m  # meters
        
    elif CONFIG["Tapeout"] == 2:
        '''
        Configuration for BTO tapeout 2 data
        '''
        if CONFIG["ring_number"] in [3,34,38,44,59]:
            CONFIG["lower_wl_limit"] = 1480
            CONFIG["upper_wl_limit"] = 1570
        # define the ring radius for run 2
        if CONFIG["ring_number"] in range(0,10) or CONFIG["ring_number"] in range(20,30) or CONFIG["ring_number"] in range(40,50):
            Ring_Rad = 50
        else:
            Ring_Rad = 80
        print(f"Ring Radius = {Ring_Rad}")
        ring_radius_m = Ring_Rad * 1e-6  # um to meters
        ring_length = 2 * np.pi * ring_radius_m  # meters
        
        base_path = r'C:\Users\Evan\Documents\ECE Masters\BTO Project\ShEtch Tapeout 2\sweepLaser'
        
        base_path = r"C:\Users\Evan\Documents\ECE Masters\BTO Project\BTO ShEtch\ShEtch Tapeout 2\sweepLaser"
        if CONFIG["Sweep_Data"] == 'Coarse':
            csv_path, folder_name = find_csv_for_ring(base_path + r"\sweepLaser", CONFIG["ring_number"])
            if csv_path:
                print(f"CSV file path: {csv_path}")    
            figures_dir = os.path.join(base_path, "RingResults_coarse", folder_name)
        elif CONFIG["Sweep_Data"] == 'Fine':
            csv_path, folder_name = find_csv_for_ring(base_path + r"\finesweepLaser", CONFIG["ring_number"])
            if csv_path:
                print(f"CSV file path: {csv_path}")
            figures_dir = os.path.join(base_path, "RingResults", folder_name)
        
    elif CONFIG["Tapeout"] == 3:
        '''
        Configuration for BTO tapeout 3 data
        '''
        base_path = r'C:\Users\Evan\Documents\ECE Masters\BTO Project\BTO ShEtch\X Chip 3\sweepLaser'
        csv_path, folder_name = find_csv_for_ring(base_path + r"\finesweepLaser", CONFIG["ring_number"])
        if csv_path:
            print(f"CSV file path: {csv_path}")    
        figures_dir = os.path.join(base_path, "RingResults_fine", folder_name)
        
    else:
        print("Sweep_Data specified wrong!!!")
            
    os.makedirs(figures_dir, exist_ok=True)
    
    return csv_path, folder_name, figures_dir, ring_radius_m, ring_length

def plot_measurements():
    fig, axs = plt.subplots(4, 2, figsize=(10, 10))

    # --- Row 1 ---
    axs[0, 0].plot(Res_df["wavelength"], Res_df["Q"], marker='o', color='purple')
    style_plot(axs[0, 0], 'Q Factor vs Wavelength', None, 'Q Factor')

    axs[0, 1].plot(fsr_centers_m * 1e9, fsr_values_m * 1e9, marker='s', color='blue')
    style_plot(axs[0, 1], 'FSR vs Wavelength', None, 'FSR (nm)')

    # --- Row 2 ---
    axs[1, 0].plot(Res_df["wavelength"], Res_df["ER_dB"], marker='s', color='gray')
    style_plot(axs[1, 0], 'ER vs Wavelength', None, 'ER (dB)')

    # axs[1, 1].plot(Measured_Details["Wvl"], Measured_Details["Prop_loss_dbcm"], marker='s',color='red')
    # axs[1, 1].plot(Measured_Details["Wvl"], Measured_Details["Coup_loss_dbcm"], marker='o',color='orange')
    # style_plot(axs[1, 1], r'$\alpha$ vs Wavelength', None, 'Loss (dB/cm)')

    # --- Row 3 ---
    axs[2, 0].plot(Res_df["wavelength"], Measured_Details["finesse"], marker='o', color='black')
    style_plot(axs[2, 0], 'Finesse vs Wavelength', None, 'Finesse')

    axs[2, 1].plot(Res_df["wavelength"], Res_df["FWHM"], marker='o', color='green')
    style_plot(axs[2, 1], 'FWHM vs Wavelength', None, 'FWHM (nm)')

    # --- Row 4 ---
    axs[3, 0].plot(fsr_centers_m * 1e9, Measured_Details["ng"], marker='d', color='magenta')
    style_plot(axs[3, 0], 'Group Index vs Wavelength', 'Wavelength (nm)', r'$n_g$')

    axs[3, 1].plot(fsr_centers_m * 1e9, Measured_Details["vg"], marker='^', color='cyan')
    style_plot(axs[3, 1], 'Group Velocity vs Wavelength', 'Wavelength (nm)', r'$v_g$ (m/s)')

    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "Combined_Q_FSR_ER_FWHM_ng_vg_finesse_alpha_vs_wavelength.png"))
    plt.show() if CONFIG["Show_Other_Plots"] else plt.close()

    # 1/finesse and 1/ER vs wavelength
    plt.figure(figsize=(8, 5))
    ax = plt.gca()
    ax.plot(Measured_Details["Wvl"], 1 / Measured_Details["finesse"], 'ro', label='1/finesse')
    ax.plot(Measured_Details["Wvl"], 1 / Res_df["ER_lin"], 'bs', markerfacecolor='none', label='1/ER')
    style_plot(ax, '1/finesse & 1/ER vs Wavelength', 'Wavelength (nm)', 'ratio')
    ax.legend()
    plt.tight_layout()
    plt.show() if CONFIG["Show_Other_Plots"] else plt.close()

    # root1 and root2 vs wavelength
    plt.figure(figsize=(8, 5))
    ax = plt.gca()
    if i_ring == 38:
        ax.plot(Measured_Details["Wvl"], root1_values, 'co', label=r'$\alpha$')
        ax.plot(Measured_Details["Wvl"], root2_values, 'bs', label=r'$t$')
        # ax.plot(Measured_Details["Wvl"], sigma1_values, marker='o',color='orange', label=r'$\kappa$_1')
        ax.plot(Measured_Details["Wvl"], sigma2_values, marker='s',color='red', label=r'$\kappa$')
        style_plot(ax, r'$\alpha$, $t$, $\kappa$ vs Wavelength', 'Wavelength (nm)', 'Coupling or Loss')
    else:
        ax.plot(Measured_Details["Wvl"], root1_values, 'co', label='root1')
        ax.plot(Measured_Details["Wvl"], root2_values, 'bs', label='root2')
        ax.plot(Measured_Details["Wvl"], sigma1_values, marker='o',color='orange', label=r'$\sigma$_1')
        ax.plot(Measured_Details["Wvl"], sigma2_values, marker='s',color='red', label=r'$\sigma$_2')
        style_plot(ax, 'root1 and root2 vs Wavelength', 'Wavelength (nm)', 'Coupling or Loss')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "root1_root2_vs_wavelength.png"))
    plt.show() if CONFIG["Show_Other_Plots"] else plt.close()
    
    plt.figure(figsize=(8, 5))
    
    FE_ovc = sigma2_values/(1-root1_values*root2_values)
    FE_unc = sigma1_values/(1-root1_values*root2_values)
    
    ax = plt.gca()
    if i_ring == 38:
        ax.plot(Measured_Details["Wvl"], FE_ovc, marker='o',color='orange', label='Field Enhancement Factor (overcoupled)')
        # ax.plot(Measured_Details["Wvl"], FE_unc, marker='o',color='navy', label='Field Enhancement Factor (undercoupled)')
    else:
        ax.plot(Measured_Details["Wvl"], FE_ovc, marker='o',color='orange', label='Field Enhancement Factor (overcoupled)')
        ax.plot(Measured_Details["Wvl"], FE_unc, marker='o',color='navy', label='Field Enhancement Factor (undercoupled)')
    ax.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
        
    # --- Plot FSR in frequency domain ---
    plt.figure(figsize=(8, 5))
    ax = plt.gca()

    # Convert wavelength (nm → m) and FSR (m) → frequency domain (Hz)
    wl_m = fsr_centers_m  # center wavelength in meters
    c = 3e8  # speed of light in m/s
    freq_Hz = c / wl_m  # frequency in Hz
    fsr_Hz = (c / (wl_m - fsr_values_m/2)) - (c / (wl_m + fsr_values_m/2))  # frequency spacing

    # Convert to THz for readability
    freq_THz = freq_Hz / 1e12
    fsr_GHz = fsr_Hz / 1e9

    ax.plot(freq_THz, fsr_GHz, marker='o', color='teal')
    ymin, ymax = ax.get_ylim()
    ax.vlines(x=1e-12*c/(Measured_Details["Wvl"][5]*1e-9),ymin=ymin,ymax=ymax,colors='gray',linestyles='--')
    style_plot(ax, 'FSR vs Frequency', 'Frequency (THz)', 'FSR (GHz)')

    plt.tight_layout()
    # plt.savefig(os.path.join(figures_dir, "FSR_vs_frequency.png"))
    plt.show() if CONFIG["Show_Other_Plots"] else plt.close()
    
    if i_ring == 38:
        # Q factor
        # alpha
        # t
        # kappa
        fig, axs = plt.subplots(3,1, figsize=(10, 8))

        # --- Row 1 ---
        axs[0].plot(Res_df["wavelength"], Res_df["Q"], marker='o', color='purple',linestyle=':')
        for i, txt in enumerate(Res_df["Q"]):
            if i in [1,3,5]:
                axs[0].text(Res_df["wavelength"][i], Res_df["Q"][i]*0.98, round(txt), ha='center', va='top', color='purple')
            if i in [2,4]:
                axs[0].text(Res_df["wavelength"][i], Res_df["Q"][i]*0.98, round(txt), ha='center', va='top', color='purple',weight='bold')
        style_plot(axs[0], 'Q Factor vs Wavelength', None, 'Q Factor')

        # --- Row 2 ---
        axs[1].plot(Measured_Details["Wvl"], root1_values, marker='o',color='orange', label=r'$\alpha$',linestyle=':')
        axs[1].plot(Measured_Details["Wvl"], root2_values, marker='s',color='blue', label=r'$t$',linestyle=':')
        axs[1].plot(Measured_Details["Wvl"], sigma2_values, marker='^',color='red', label=r'$\kappa$',linestyle=':')
        for i, txt in enumerate(root1_values):
            if i in [1,3,5]:
                axs[1].text(Measured_Details["Wvl"][i], root1_values[i]*0.96, round(root1_values[i],3), ha='center', va='top', color='orange')
                axs[1].text(Measured_Details["Wvl"][i], root2_values[i]*0.935, round(root2_values[i],3), ha='center', va='top', color='blue')
                axs[1].text(Measured_Details["Wvl"][i], sigma2_values[i]*1.02, round(sigma2_values[i],3), ha='center', va='bottom', color='red')
            if i in [2,4]:
                axs[1].text(Measured_Details["Wvl"][i], root1_values[i]*0.96, round(root1_values[i],3), ha='center', va='top', color='orange',weight='bold')
                axs[1].text(Measured_Details["Wvl"][i], root2_values[i]*0.935, round(root2_values[i],3), ha='center', va='top', color='blue',weight='bold')
                axs[1].text(Measured_Details["Wvl"][i], sigma2_values[i]*1.02, round(sigma2_values[i],3), ha='center', va='bottom', color='red',weight='bold')
                
        style_plot(axs[1], r'$\alpha$, $t$, $\kappa$ vs Wavelength', None, 'Coupling or Loss')

        # Row 3
        axs[2].plot(Measured_Details["Wvl"], FE_ovc, marker='o',color='navy', label='Field Enhancement Factor (overcoupled)',linestyle=':')
        for i, txt in enumerate(FE_ovc):
            if i in [1,3,5]:
                axs[2].text(Measured_Details["Wvl"][i], FE_ovc[i]*0.99, round(txt,3), ha='center', va='top', color='navy')
            if i in [4]:
                axs[2].text(Measured_Details["Wvl"][i], FE_ovc[i]*0.99, round(txt,3), ha='center', va='top', color='navy',weight='bold')
            if i in [2]:
                axs[2].text(Measured_Details["Wvl"][i], FE_ovc[i]*1.01, round(txt,3), ha='center', va='top', color='navy',weight='bold')
        style_plot(axs[2], 'Field Enhancement Factor', 'Wavelength (nm)', 'FEF')
        axs[1].legend()

        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, "Passive_Ring_Params.png"))
        plt.show() if CONFIG["Show_Other_Plots"] else plt.close()
        
        pass
    
    return

def save_data():
    # Save to CSV
    resonance_data = pd.DataFrame({
        'Resonance Wavelength (nm)': Res_df["wavelength"],
        'Q Factor': Res_df["Q"],
        'FWHM (nm)': Res_df["FWHM"],
        'ER (dB)': Res_df["ER_dB"],
        'root1': root1_values,
        'root2': root2_values,
    })
    print(len(Measured_Details["R_sqr"]))
    print(len(Measured_Details["ng"]))
    group_data = pd.DataFrame({
        'Resonance Wavelength (nm)': fsr_centers_m*1e9,
        'Group index': Measured_Details["ng"],
        'Group Velocity': Measured_Details["vg"],
        'R_Squared': Measured_Details["R_sqr"][:-1],
    })
    csv_path = os.path.join(figures_dir, 'Resonance_Data.csv')
    resonance_data.to_csv(csv_path, index=False)
    print(f"Saved resonance data to {csv_path}")
    
    csv_path = os.path.join(figures_dir, 'Group_Data.csv')
    group_data.to_csv(csv_path, index=False)
    print(f"Saved group data to {csv_path}")
    return

num_rings = 1
rngs = [38]

for i_ring in rngs:
    if not(i_ring in [3]):
        if 'closest_resonance' in locals():
            del closest_resonance
        
        # ==== USER PARAMETERS ====
        CONFIG = {
            "lower_wl_limit": 1460,     # nm
            "upper_wl_limit": 1580,     # nm
            "fit_window_nm": 0.6,       # window size in nm around each peak for fitting
            "num_edge_points": 3,       # number of edge points for Tmax calculation
            "res_ER_thrshld": 1.0,      # dBm (ER required to identify a resonance)
            "Show_Resonance_Plots": False, # show fitting plot for each resonance 
            "Show_Other_Plots": True, # show the other plots (spectrum, other ring params ect..)
            "ring_number": i_ring, # if you are iterating over multiple rings
            "Flatten_Spectrum": True,
            "Subtract_GC": False, # if you have an exact GC response to subtract
            "Sweep_Data": 'Fine', # for choosing between tapeout 1 and 2 fine/coarse data
            "Tapeout": 2, #change this if you are pointing to new data
        }
        # ==========================
        csv_path, folder_name, figures_dir, ring_radius_m, ring_length = configure()

        # === Main Processing ===

        ring_wl, ring_frq, ring_spec = load_data_from_specific_lines(csv_path, dvc='ring')
        in_range_peaks, flattened_spec = Plot_and_Save_Spectra(ring_wl,ring_frq,ring_spec,figures_dir)

        # === Analyze Resonances ===

        Resonance_Details = []
        Measured_Details = {}

        for i, peak in enumerate(in_range_peaks):
            # fit lorentzian 
            opt_lorentz_params, x_data, y_data_mW, pcov = utils.fit_lorentzian_wvl(ring_wl, flattened_spec, peak, window_nm=CONFIG["fit_window_nm"])
            
            _,_,_,_ = utils.fit_lorentzian_frq(ring_frq, ring_spec, peak, window_THz=0.1)
            
            lorentz_params_err = np.sqrt(np.diag(pcov))
            if opt_lorentz_params is not None:
                A, lam0, fwhm, offset = opt_lorentz_params
                A_err, lam0_err, FWHM_err, offset_err = lorentz_params_err
                
                # Convert y_data to dBm scale
                y_data_dBm = 10*np.log10(y_data_mW)
                
                # Calculate & save the Q and FWHM for each resonance
                print(f'   λ_res = {lam0:.2f} nm')
                # fwhm, Q = calculate_q_fwhm(lam0, gamma)
                Q = utils.Calculate_Q(lam0, fwhm)
                print(f"----> {Q}")
                    
                    
                Q_err = Q * np.sqrt((lam0_err / lam0)**2 + (FWHM_err / fwhm)**2)
                print(f'   Q = {Q:.2f} +- {Q_err:.2f}')
                # fwhm_err = 2 * gamma_err
                print(f'   FWHM = {fwhm:.3f} +- {FWHM_err:.3f} nm')
                
                # Generate lorentzian spectrum and calculate R² in mW 
                y_fit_mW= utils.Lorentzian(x_data, *opt_lorentz_params)
                r_squared = calculate_r_squared(y_data_mW, y_fit_mW)
                print(f"   r_sqr = {round(r_squared,4)}")

                # Smooth out the discretized lorentzian for plotting
                x_smooth = np.linspace(x_data.min(), x_data.max(), 500)
                fit_y_mW_smooth = utils.Lorentzian(x_smooth, *opt_lorentz_params)
                fit_y_dBm_smooth = 10*np.log10(fit_y_mW_smooth)

                if len(y_data_dBm) >= 2 * CONFIG["num_edge_points"]:
                    edge_values = np.concatenate([y_data_dBm[:CONFIG["num_edge_points"]], y_data_dBm[-CONFIG["num_edge_points"]:]])
                else:
                    edge_values = np.concatenate([y_data_dBm[:1], y_data_dBm[-1:]])
                    
                # Calculate Tmax from the average of the resonance edge points, and Tmin from resonance minimum
                Tmax = np.mean(edge_values)
                #Tmax = np.max(y_data_dBm)
                #Tmax = 0
                #Tmin = np.min(y_data_dBm)
                Tmin = np.min(fit_y_dBm_smooth)

                # Calculate the extinction ratio
                ER_dB = Tmax - Tmin
                print(f"   Tmax dB: {Tmax}, Tmax lin: {10**(Tmax/10)}")
                print(f"   Tmin dB: {Tmin}, Tmin lin: {10**(Tmin/10)}")
                print(f'   ER = {ER_dB:.3f} dB')
                print(f'   Ring # = {i_ring}')
                ER_linear = 10**(ER_dB/10)
                
                Tmin_lin = 10**(((1-Tmax) + Tmin)/10)
                
                # Loaded, intrinsic and external Q factor
                Qc_und = 2 * Q / (1 - np.sqrt(Tmin_lin))
                Qc_ovr = 2 * Q / (1 + np.sqrt(Tmin_lin))
                
                Qi_und = Qc_und * Q / (Qc_und - Q)
                Qi_ovr = Qc_ovr * Q / (Qc_ovr - Q)
                
                print(f"--> Undercoupled: Qc = {Qc_und}, Qi = {Qi_und}")
                print(f"--> Overcoupled: Qc = {Qc_ovr}, Qi = {Qi_ovr}")
                    
                Resonance_Details.append({
                    "ring": i_ring,
                    "peak_index": i,
                    "wavelength": lam0,
                    "Q": Q,
                    "Q_err": Q_err,
                    "FWHM": fwhm,
                    "fwhm_err": FWHM_err,
                    "ER_dB": ER_dB,
                    "ER_lin": ER_linear,
                    "R_sqr": r_squared
                })
                
                Plot_and_Save_Resonance(in_range_peaks, x_data, y_data_dBm, x_smooth, fit_y_dBm_smooth, Resonance_Details, figures_dir)

            else:
                print(f"Could not fit Lorentzian to resonance {i+1}")

            Res_df = pd.DataFrame(Resonance_Details)
                
            # Save resonance closest to 1550 nm
            target_wl = 1550  # target resonance
            if 'closest_resonance' not in locals():
                closest_resonance = {'index': i, 'wl_diff': abs(lam0 - target_wl)}
            elif abs(lam0 - target_wl) < closest_resonance['wl_diff']:
                closest_resonance = {'index': i, 'wl_diff': abs(lam0 - target_wl)}

                # Offset wavelength so resonance is at 0
                wavelength_offset = x_data - lam0  # offset from resonance peak

                # Save the data for this closest resonance
                df_resonance = pd.DataFrame({
                    'Wavelength Offset (nm)': wavelength_offset,
                    'Spectrum (dBm)': y_data_dBm,
                    'Lorentzian Fit (dBm)': 10 * np.log10(utils.Lorentzian(x_data, *opt_lorentz_params)),
                })
                
                # Save to CSV
                save_path = os.path.join(figures_dir, f"resonance_closest_to_{target_wl}nm.csv")
                df_resonance.to_csv(save_path, index=False)

        # === FSR and Derived Plots ===
        # Calculate FSR from resonance wavelength diff
        # fsr_values = []
        if len(Res_df["wavelength"]) > 1:
            Measured_Details["Wvl"] = Res_df["wavelength"].values
            Measured_Details["R_sqr"] = Res_df["R_sqr"].values
            
            fsr_values_m = np.diff(Measured_Details["Wvl"]) * 1e-9
            fsr_centers_m = 0.5e-9 * (Measured_Details["Wvl"][:-1] + Measured_Details["Wvl"][1:])
            fsr_interp_m = np.interp(Measured_Details["Wvl"], fsr_centers_m, fsr_values_m) if len(fsr_values_m) > 0 else np.zeros_like(Measured_Details["Wvl"])
            
            #Q_array = np.array(Q_values)

            # === Calculate Group Index ===
            
            Measured_Details["fsr_interp"] = fsr_interp_m
            
            # Group index: ng = λ² / (FSR × L)
            Measured_Details["ng"] = (fsr_centers_m ** 2) / (fsr_values_m * ring_length)
            
            # === Calculate Group Velocity ===
            c = 3e8  # speed of light in m/s
            Measured_Details["vg"] = c / Measured_Details["ng"]  # m/s

            # === Calculate Group Velocity Dispersion (GVD) ===
            # Convert fsr_centers to meters
            wavelengths_m = fsr_centers_m

            # Take second derivative of ng vs λ
            ng_spline = UnivariateSpline(wavelengths_m, Measured_Details["ng"], k=3, s=0)
            d2ng_dlambda2 = ng_spline.derivative(n=2)(wavelengths_m)

            # GVD in units of s^2/m
            beta2 = - (wavelengths_m ** 3) / (2 * np.pi * c**2) * d2ng_dlambda2
            
            Measured_Details["finesse"] = fsr_interp_m*1e9 / np.array(Res_df["FWHM"])
        
            root1_values, root2_values, sigma1_values, sigma2_values = Calculate_roots(Measured_Details["finesse"], Res_df["ER_lin"])
            # alpha vs wavelength (propagation loss)
            # Measured_Details["Prop_loss_dbcm"],Measured_Details["Coup_loss_dbcm"] = Calculate_Prop_Loss(Res_df["ER_lin"],Measured_Details["finesse"],ring_radius_m,)

        plot_measurements()
        save_data()

        

