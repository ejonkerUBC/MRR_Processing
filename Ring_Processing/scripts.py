# scripts.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import utils

def plot_resonance(x_data, y_data_dBm, x_smooth, fit_y_dBm_smooth, ResSpcs, DoPlot=False):
    # Plot each resonance
    fig = plt.figure(figsize=(8, 4))
    plt.plot(x_data-ResSpcs[-1]["wavelength"], y_data_dBm, 'b.', label='Data')
    plt.plot(x_smooth-ResSpcs[-1]["wavelength"], fit_y_dBm_smooth, 'r-', label='Lorentzian Fit')
    plt.axvline(0, color='g', linestyle='--', label=f'Resonance = {ResSpcs[-1]["wavelength"]:.2f} nm')
    plt.xlabel('Wavelength detuning (nm)'); plt.ylabel('Power (dBm)'); plt.ticklabel_format(useOffset=False)
    plt.title(f'Resonance {ResSpcs[-1]["peak_index"]+1}\nFWHM = {ResSpcs[-1]["FWHM"]:.4f}+-{ResSpcs[-1]["FWHM_err"]:.4f} nm,   Q = {ResSpcs[-1]["Q"]:.2f}+-{ResSpcs[-1]["Q_err"]:.2f},   ER = {ResSpcs[-1]["ER_dB"]:.3f} dBm')
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.show() if DoPlot else plt.close()
    return fig

def style_plot(ax, title, xlabel, ylabel):
    '''
    Function to simplify plot formatting
    '''
    ax.set_title(title) if title is not None else None
    ax.set_xlabel(xlabel) if xlabel is not None else None
    ax.set_ylabel(ylabel) if ylabel is not None else None
    ax.grid(True)
    return

def plot_measurements(Res_df,Measured_Details,fsr_centers_m,fsr_values_m,save_dir):
    ''' plots all ring parameters in a single figure
        ________________
        Qfactor |   FSR
        ER      |   ??
        Finesse |   FWHM
        ng      |   vg
        ________|_______
    '''
    # Figure
    fig, axs = plt.subplots(4, 2, figsize=(10, 10))

    # --- Row 1 ---
    axs[0, 0].plot(Res_df["wavelength"], Res_df["Q"], marker='o', color='purple')
    style_plot(axs[0, 0], 'Q Factor vs Wavelength', None, 'Q Factor')

    axs[0, 1].plot(fsr_centers_m * 1e9, fsr_values_m * 1e9, marker='s', color='blue')
    style_plot(axs[0, 1], 'FSR vs Wavelength', None, 'FSR (nm)')

    # --- Row 2 ---
    axs[1, 0].plot(Res_df["wavelength"], Res_df["ER_dB"], marker='s', color='gray')
    style_plot(axs[1, 0], 'ER vs Wavelength', None, 'ER (dB)')

    axs[1, 1].plot(Res_df["wavelength"], Measured_Details["GVD"]*1e20, marker='s',color='red')
    style_plot(axs[1, 1], 'GVD', None, r'$\beta2$ ($ps^2/m$)')

    # --- Row 3 ---
    axs[2, 0].plot(Res_df["wavelength"], Measured_Details["Finesse"], marker='o', color='black')
    style_plot(axs[2, 0], 'Finesse vs Wavelength', None, 'Finesse')

    axs[2, 1].plot(Res_df["wavelength"], Res_df["FWHM"], marker='o', color='green')
    style_plot(axs[2, 1], 'FWHM vs Wavelength', None, 'FWHM (nm)')

    # --- Row 4 ---
    axs[3, 0].plot(Res_df["wavelength"], Measured_Details["ng"], marker='d', color='magenta')
    style_plot(axs[3, 0], 'Group Index vs Wavelength', 'Wavelength (nm)', r'$n_g$')

    axs[3, 1].plot(Res_df["wavelength"], Measured_Details["vg"], marker='^', color='cyan')
    style_plot(axs[3, 1], 'Group Velocity vs Wavelength', 'Wavelength (nm)', r'$v_g$ (m/s)')

    plt.tight_layout()
    plt.show()
    fig.savefig(os.path.join(save_dir, 'all_measurements.png'), dpi=300)
    return

def roots_vs_wavelength(Measured_Details,ER_lin_array,save_dir):
    '''
    Script to calculate and plot root1 and root2 (alpha and tau) vs wavelength.
    '''
    root1_values, root2_values, kappa1_values, kappa2_values = utils.Calculate_Roots(Measured_Details["Finesse"], ER_lin_array)
    
    # root1 and root2 vs wavelength
    fig = plt.figure(figsize=(8, 5))
    ax = plt.gca()
    ax.plot(Measured_Details["Wvl"], root1_values, 'co', label=r'$root_1$')
    ax.plot(Measured_Details["Wvl"], root2_values, 'bs', label=r'$root_2$')
    ax.plot(Measured_Details["Wvl"], kappa1_values, marker='o',color='orange', label=r'$\kappa_1$')
    ax.plot(Measured_Details["Wvl"], kappa2_values, marker='s',color='red', label=r'$\kappa_2$')
    style_plot(ax, r'$\alpha$, $t$, $\kappa$ vs Wavelength', 'Wavelength (nm)', 'Coupling or Loss')
    ax.legend()
    plt.tight_layout()
    plt.show()
    fig.savefig(os.path.join(save_dir, 'roots_vs_wvl.png'), dpi=300)
    return fig

def ER_Finesse_vs_wvl(Measured_Details,Res_df):
    # 1/finesse and 1/ER vs wavelength
    fig = plt.figure(figsize=(8, 5))
    ax = plt.gca()
    ax.plot(Measured_Details["Wvl"], 1 / Measured_Details["Finesse"], 'ro', label='1/finesse')
    ax.plot(Measured_Details["Wvl"], 1 / Res_df["ER_lin"], 'bs', markerfacecolor='none', label='1/ER')
    style_plot(ax, '1/finesse & 1/ER vs Wavelength', 'Wavelength (nm)', 'ratio')
    ax.legend()
    plt.tight_layout()
    plt.show()
    return

def Field_Enhancement_Factor(Measured_Details,ER_lin_array):
    root1_values, root2_values, kappa1_values, kappa2_values = utils.Calculate_Roots(Measured_Details["Finesse"], ER_lin_array)
    
    fig = plt.figure(figsize=(8, 5))
    
    FE_ovc = kappa2_values/(1-root1_values*root2_values)
    FE_unc = kappa1_values/(1-root1_values*root2_values)
    
    ax = plt.gca()
    ax.plot(Measured_Details["Wvl"], FE_ovc, marker='o',color='orange', label='Field Enhancement Factor (overcoupled?)')
    ax.plot(Measured_Details["Wvl"], FE_unc, marker='o',color='navy', label='Field Enhancement Factor (undercoupled?)')

    ax.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    return fig

def FSR_Freq(fsr_centers_m,fsr_values_m,Measured_Details):
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
    plt.show()
    
def Analyze_Resonances(in_range_peaks,ring_wl,flattened_spec,save_dir,DoPlot=False):
    '''
    Script to fit data with lorentzian and extract relevant parameters: Q, FWHM, ER and errors
    '''
    Resonance_Details = []

    figures_dir = os.path.join(save_dir, "Resonances")
    os.makedirs(figures_dir, exist_ok=True)
    os.chdir(figures_dir)

    for i, peak in enumerate(in_range_peaks):
        # fit lorentzian 
        opt_lorentz_params, x_data, y_data_mW, pcov = utils.fit_lorentzian_wvl(ring_wl, flattened_spec, peak)
        
        # optionally fit in freq space
        # _,_,_,_ = utils.fit_lorentzian_frq(ring_frq, ring_spec, peak, window_THz=0.1)
        
        lorentz_params_err = np.sqrt(np.diag(pcov))
        if opt_lorentz_params is not None:
            A, lam0, fwhm, offset = opt_lorentz_params
            A_err, lam0_err, FWHM_err, offset_err = lorentz_params_err
            
            # Convert y_data to dBm scale
            y_data_dBm = 10*np.log10(y_data_mW)
            
            # Calculate & save the Q and FWHM for each resonance
            Q = utils.Calculate_Q(lam0, fwhm)
            Q_err = Q * np.sqrt((lam0_err / lam0)**2 + (FWHM_err / fwhm)**2)

            # fwhm_err = 2 * gamma_err

            # Generate lorentzian spectrum and calculate R² in mW 
            # y_fit_mW= utils.Lorentzian(x_data, *opt_lorentz_params)

            # Smooth out the discretized lorentzian for plotting
            x_smooth = np.linspace(x_data.min(), x_data.max(), 500)
            fit_y_mW_smooth = utils.Lorentzian(x_smooth, *opt_lorentz_params)
            fit_y_dBm_smooth = 10*np.log10(fit_y_mW_smooth)

            num_edge_pts = 3

            if len(y_data_dBm) >= 2 * num_edge_pts:
                edge_values = np.concatenate([y_data_dBm[:num_edge_pts], y_data_dBm[-num_edge_pts:]])
            else:
                edge_values = np.concatenate([y_data_dBm[:1], y_data_dBm[-1:]])
                
            # Calculate Tmax from the average of the resonance edge points, and Tmin from resonance minimum
            Tmax = np.mean(edge_values)
            Tmin = np.min(fit_y_dBm_smooth)
            Tmin_lin = 10**(((1-Tmax) + Tmin)/10)

            # Calculate the extinction ratio
            ER_dB = Tmax - Tmin
            ER_linear = 10**(ER_dB/10)
            
            # Loaded, intrinsic and external Q factor
            Qc_und = 2 * Q / (1 - np.sqrt(Tmin_lin))
            Qc_ovr = 2 * Q / (1 + np.sqrt(Tmin_lin))
            
            Qi_und = Qc_und * Q / (Qc_und - Q)
            Qi_ovr = Qc_ovr * Q / (Qc_ovr - Q)
            
            # save data to dictionary
            Resonance_Details.append({
                "peak_index": i,
                "wavelength": lam0,
                "Q": Q,
                "Q_err": Q_err,
                "FWHM": fwhm,
                "FWHM_err": FWHM_err,
                "ER_dB": ER_dB,
                "ER_lin": ER_linear,
            })
            
            # summarize results
            print(f"Resonance {Resonance_Details[-1]["peak_index"]+1}/{len(in_range_peaks)}:")
            print(f'   λ_res = {lam0:.2f} nm')
            print(f'   Q = {Q:.2f} +- {Q_err:.2f}')
            print(f"        if Undercoupled: Qc = {Qc_und}, Qi = {Qi_und}")
            print(f"        if Overcoupled: Qc = {Qc_ovr}, Qi = {Qi_ovr}")
            print(f'   FWHM = {fwhm:.3f} +- {FWHM_err:.3f} nm')
            print(f"   Tmax dB: {Tmax}, Tmax lin: {10**(Tmax/10)}")
            print(f"   Tmin dB: {Tmin}, Tmin lin: {10**(Tmin/10)}")
            print(f'   ER = {ER_dB:.3f} dB')

            fig = plot_resonance(x_data, y_data_dBm, x_smooth, fit_y_dBm_smooth, Resonance_Details, DoPlot=DoPlot)
            fig.savefig(f"Res{i}.png", dpi=300)
        else:
            print(f"Could not fit Lorentzian to resonance {i+1}")
        
        Res_df = pd.DataFrame(Resonance_Details)
        resonance_data = pd.DataFrame({
            'Resonance Wavelength (nm)': Res_df["wavelength"],
            'Q Factor': Res_df["Q"],
            'FWHM (nm)': Res_df["FWHM"],
            'ER (dB)': Res_df["ER_dB"],
            })
        csv_path = os.path.join(save_dir, 'Resonance_analysis_data.csv')
        resonance_data.to_csv(csv_path, index=False)
        print(f"Saved resonance data to {csv_path}")
        print(" ")

    return Res_df