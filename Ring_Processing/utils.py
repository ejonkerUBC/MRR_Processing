# utils.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline
import os

def Calculate_Q(lam0, FWHM):
    '''
    Calculate the Q-factor from FWHM and central wavelength.
        FWHM: Full width half-max
        lam0: central wavelength
    '''
    FWHM = abs(FWHM)
    Q = lam0 / FWHM if FWHM != 0 else np.inf
    return Q

def Lorentzian(x, A, x0, FWHM, offset):
    '''
    Lorentzian function definition.
        A: amplitude
        FWHM: Full width half-max
        lam0: central wavelength
        offset: y-displacement
    '''
    return offset - (A * ((FWHM/2)**2) / ((x - x0)**2 + (FWHM/2)**2))

def Calculate_Roots(finesse_array, ER_lin_array):
    ''' 
    Calculate root1, root2, kappa1, and kappa2. {root1,root2} = {alpha,t}, but are 
    indestinguishable from the calculation. kappa1 or kappa2 is the cross coupling 
    coefficient calculated from t.
        finesse_array: array of resonance finesse
        ER_lin_array: array of linear extinction ratios
    '''
    A_values = np.cos(np.pi / finesse_array) / (1 + np.sin(np.pi / finesse_array))
    B_values = 1 - (1 / ER_lin_array) * ((1 - np.cos(np.pi / finesse_array)) / (1 + np.cos(np.pi / finesse_array)))
    root1_values = np.sqrt(A_values / B_values) + np.sqrt((A_values / B_values) - A_values)
    root2_values = np.sqrt(A_values / B_values) - np.sqrt((A_values / B_values) - A_values)
    kappa1_values = np.sqrt(1-root1_values**2) #kappa1
    kappa2_values = np.sqrt(1-root2_values**2) #kappa2
    return root1_values, root2_values, kappa1_values, kappa2_values

def Calculate_Prop_Loss(r_ring, alpha_rt):
    '''
    Calculate propagation loss from round trip attenuation
        r_ring: ring radius in m
        alpha_rt: linear round trip field attenuation
    '''
    alpha_dB_m = -10*np.log10(alpha_rt**2)/(2*np.pi*r_ring)
    alpha_dB_cm = alpha_dB_m/100
    return alpha_dB_cm

def Calculate_Coupler_Loss(tau):
    '''
    Calculate coupler loss from self coupling coefficient
        t: self coupling coefficient
    '''
    tau_dB = -10*np.log10(tau**2)
    return tau_dB

def Intrinsic_and_Coupling_Q (QL,Tmin_lin):
    '''
    Calculate intrinsic and external/coupling Q-factor from the loaded Q
        QL: Loaded Q
        Tmin_lin: linear minimum transmission
    '''
    Qc_und = 2 * QL / (1 - np.sqrt(Tmin_lin))
    Qc_ovr = 2 * QL / (1 + np.sqrt(Tmin_lin))
    
    Qi_und = Qc_und * QL / (Qc_und - QL)
    Qi_ovr = Qc_ovr * QL / (Qc_ovr - QL)
    
    print(f"--> Undercoupled: Qc = {Qc_und}, Qi = {Qi_und}")
    print(f"--> Overcoupled: Qc = {Qc_ovr}, Qi = {Qi_ovr}")
    return Qc_und, Qi_und, Qc_ovr, Qi_ovr

def Group_Index(FSRs,ring_rad,fsr_centers_m):
    '''
    calculate group index from FSR, ring radius and wvl
    ng = λ² / (L x FSR)
        FSR: Free spectral range
        ring_rad: ring radius
        fsr_centers_m: wvl points midway between resonances
    '''
    ring_length = 2*np.pi*ring_rad
    ng = (fsr_centers_m ** 2) / (FSRs * ring_length)
    return ng

def Group_Velocity(ng):
    '''
    calculate group velocity from group index
    vg = c / ng
        ng: group index
    '''
    c = 299792458
    vg = c / ng 
    return vg

def fit_lorentzian_wvl(wavelength, spectrum, peak_index, window_nm=0.5):
    '''
    Extract the optimal lorentzian parameters in the mW scale using 
    optimization curve_fit function.
    
    '''
    # Determine central wavelength
    center_wl = wavelength[peak_index]
    
    # Gather the data in resonance window
    indices = np.where((wavelength >= center_wl - window_nm) & (wavelength <= center_wl + window_nm))[0]
    if len(indices) < 5:
        return None, wavelength[indices], spectrum[indices]
    x_data = wavelength[indices]
    y_data_dBm = spectrum[indices]
    
    # Convert the y data to mW scale for fitting
    y_data_mW = 10**(y_data_dBm/10)
    
    # Make parameter guesses for optimization
    A_guess = np.max(y_data_mW) - np.min(y_data_mW)
    x0_guess = center_wl
    gamma_guess = window_nm / 2
    offset_guess = np.mean(y_data_mW)
    p0 = [A_guess, x0_guess, gamma_guess, offset_guess]
    
    try:
        popt, pcov = curve_fit(Lorentzian, x_data, y_data_mW, p0=p0)
        return popt, x_data, y_data_mW, pcov
    except RuntimeError:
        return None, x_data, y_data_mW
    
def fit_lorentzian_frq(frq, spectrum, peak_index, window_THz=0.1):
    '''
    Extract the optimal lorentzian parameters in the mW scale using 
    optimization curve_fit function.
    '''
    # Determine central wavelength
    center_frq = frq[peak_index]
    
    # Gather the data in resonance window
    indices = np.where((frq >= center_frq - window_THz) & (frq <= center_frq + window_THz))[0]
    if len(indices) < 5:
        return None, frq[indices], spectrum[indices]
    x_data = frq[indices]
    y_data_dBm = spectrum[indices]
    
    # Convert the y data to mW scale for fitting
    y_data_mW = 10**(y_data_dBm/10)
    
    # Make parameter guesses for optimization
    A_guess = np.max(y_data_mW) - np.min(y_data_mW)
    x0_guess = center_frq
    gamma_guess = window_THz / 2
    offset_guess = np.mean(y_data_mW)
    p0 = [A_guess, x0_guess, gamma_guess, offset_guess]
    
    # Calculate optimal Lorentzian parameters
    try:
        popt, pcov = curve_fit(Lorentzian, x_data, y_data_mW, p0=p0)
        y_fit_mW = Lorentzian(x_data, *popt)
        A, frq0, FWHM, offset = popt
        # plt.figure(figsize=(8, 4))
        # plt.scatter((x_data - frq0)*1e3,y_data_dBm,color='orange',marker='o',facecolors='none')
        # plt.plot((x_data - frq0)*1e3,10*np.log10(y_fit_mW),color='navy',linestyle='-')
        # plt.xlabel('Frequency Detuning (GHz)')
        # plt.ylabel('Power (dBm)')
        
        # plt.grid(True)
        # plt.suptitle(f"Resonance {round(1e-3*299792458/frq0,3)}")
        # plt.title(f"Q = {round(frq0/FWHM,2)}", color='gray')
        # plt.show()
        
        return popt, x_data, y_data_mW, pcov
    except RuntimeError:

        return None, x_data, y_data_mW

import numpy as np
import pandas as pd

import numpy as np

import numpy as np

def load_data_scylla_csv(file_path, wavelength_key=None, spectrum_key=None):
    '''
    Extract the wavelength and spectrum data from specified .csv file assuming Scylla formatting.
    Function will search for common wavelength/spectrum data labels if none are provided and output 
    wavelength in m, frequency in Hz, and the spectrum in the units of the raw data.
    '''
    
    wavelength = None
    spectrum = None

    # Default candidate keys (case-insensitive)
    wvl_keys = ([wavelength_key] if wavelength_key is not None
        else ['wavelength', 'wvl', 'wavelength (nm)'])
    spec_keys = ([spectrum_key] if spectrum_key is not None
        else ['spectrum', 'spec', 'channel_0', 'channel_1', 'channel_2', 'channel_3', 'channel_4'])

    # Normalize keys for matching
    wvl_keys = [k.lower() for k in wvl_keys]
    spec_keys = [k.lower() for k in spec_keys]

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(',')

            if len(parts) < 2:
                continue

            row_key = parts[0].strip().lower()

            # Only parse data AFTER a key match
            if wavelength is None and row_key in wvl_keys:
                wavelength = np.array(parts[1:], dtype=float)

            elif spectrum is None and row_key in spec_keys:
                spectrum = np.array(parts[1:], dtype=float)

            # Early exit if both found
            if wavelength is not None and spectrum is not None:
                break

    if wavelength is None:
        raise ValueError(f"No wavelength row found (tried {wvl_keys})")

    if spectrum is None:
        raise ValueError(f"No spectrum row found (tried {spec_keys})")

    # Ensure wavelength is in m
    if np.mean(wavelength) > 1e-6:
        wavelength_m = wavelength * 1e-9
    else:
        wavelength_m = wavelength

    # Convert wavelength → frequency
    c = 299792458  # m/s
    frequency_Hz = c / wavelength_m

    return wavelength_m, frequency_Hz, spectrum

def Find_Resonance_Idxs(ring_wl_m, ring_spec, save_dir, wvl_range=[], FSR_m=None, DoPlot=False):
    '''
    Find_Resonance_Idxs finds the indecies of the resonances from a given spectrum. 
    Arrays of wavelength and spectrum data are required, and a wavelength range can 
    be specified to restrict where the function finds peaks. If an estimation of the 
    FSR is given it will clarify the spacing between adjacent peaks, if none is 
    provided it will use a default value of 2nm.
    '''
    from scipy.signal import find_peaks
    
    wvl_range = [np.min(ring_wl_m),np.max(ring_wl_m)] if wvl_range == [] else wvl_range
    res_ER_thrshld = 1.0 #Threshold extinction ratio to satisfy resonance condition (dB)
    DEFAULT_SPACING = 2.0e-9 # Default minimum spacing between resonances
    min_resonance_spacing_m = DEFAULT_SPACING if FSR_m is None else 0.75 * FSR_m
    
    # Find resonances
    nm_per_point = np.mean(np.diff(ring_wl_m))
    min_distance_pts = int(min_resonance_spacing_m / nm_per_point)
    peaks, _ = find_peaks(-ring_spec, prominence=res_ER_thrshld, distance=min_distance_pts)
    in_range_pks_idx = np.array([p for p in peaks if wvl_range[0] <= ring_wl_m[p] <= wvl_range[1]])
    out_of_range_pks_idx = np.array([p for p in peaks if p not in in_range_pks_idx])
    
    in_range_pks_m = ring_wl_m[in_range_pks_idx]
    try:
        out_of_range_pks_m = ring_wl_m[out_of_range_pks_idx]
        OoR_pk_spec = ring_spec[out_of_range_pks_idx]
    except:
        out_of_range_pks_m = []
        OoR_pk_spec = []
            
    if DoPlot:
    # Plot spectrum with resonances
        fig = plt.figure(figsize=(6, 5))
        plt.plot(ring_wl_m, ring_spec, label='Spectrum',color='navy')
        plt.plot(in_range_pks_m, ring_spec[in_range_pks_idx], 'ro', label='In-Range Resonances')
        plt.plot(out_of_range_pks_m, OoR_pk_spec, 'ro', mfc='none')
        plt.axvline(wvl_range[0], color='dimgray', linestyle='--')
        plt.axvline(wvl_range[1], color='dimgray', linestyle='--')
        plt.xlim([ring_wl_m[0], ring_wl_m[-1]])
        plt.xlabel('Wavelength (nm)',fontsize=13);   plt.ylabel('Power (dBm)',fontsize=13)
        plt.legend(); 
        plt.title("Spectrum with Resonances")
        plt.grid(True); plt.tight_layout()  
        plt.show()
        fig.savefig(os.path.join(save_dir, 'Spectrum_with_Resonances.png'), dpi=300)
    return in_range_pks_idx

def Find_Background_Poly(wavelength_m, ring_spec, save_dir, FSR_m=None, DoPlot=False):
    '''
    Find_Background_Poly finds the in range peaks and interpolates the 
    background polynomial between resonances to flatten the spectrum.
    '''
    # Find resonances
    in_range_peaks = Find_Resonance_Idxs(wavelength_m, ring_spec, save_dir='', wvl_range=[1.4e-6,1.6e-6], FSR_m=FSR_m ,DoPlot=False)
    res_wvls_m = wavelength_m[in_range_peaks]  # resonance wavelengths in m
    
    try:
        if len(res_wvls_m) >= 2:
            
            # Midpoints between adjacent resonances
            mid_idxs = np.array([], dtype=int)
            for i in range(len(res_wvls_m)-1):
                midpt_wvl_m = 0.5 * (res_wvls_m[i] + res_wvls_m[i+1])
                idx = np.argmin(np.abs(wavelength_m - midpt_wvl_m))
                mid_idxs = np.append(mid_idxs,idx)

            # Add missing start/end points
            mid_idx_diff = np.round(np.mean(np.diff(mid_idxs)))
            
            for count in range(20):
                if len(wavelength_m) > (mid_idxs[-1] + mid_idx_diff):
                    mid_idxs = np.append(mid_idxs, np.int64(mid_idxs[-1] + mid_idx_diff))
                else:
                    break
            for count in range(20):
                if (mid_idxs[0] - mid_idx_diff) > 0:
                    mid_idxs = np.insert(mid_idxs, 0, np.int64(mid_idxs[0] - mid_idx_diff))
                else:
                    break
                
            if mid_idxs.size > 0:
                spline_fit = UnivariateSpline(wavelength_m[mid_idxs],ring_spec[mid_idxs],s=0.01)
                baseline = spline_fit(wavelength_m)
                flattened_spec = ring_spec - baseline
            else:
                flattened_spec = ring_spec
                print("No midpoint indices found")

        else:
            flattened_spec = ring_spec
            print("Less than 2 resonances")

    except Exception as e:
        print(f"Baseline correction failed: {e}")
        flattened_spec = ring_spec

    if DoPlot:
        fig = plt.figure(1)
        plt.title('Flattened Spectrum')
        plt.plot(wavelength_m*1e9,flattened_spec,color='navy')
        plt.xlabel("Wavelength (nm)",fontsize=13)
        plt.ylabel("Power (dBm)",fontsize=13)
        fig.savefig(os.path.join(save_dir, 'Flattened_Spec.png'), dpi=300)
        
        fig = plt.figure(2)
        plt.title('Background Polynomial')
        plt.plot(wavelength_m*1e9,ring_spec,color='navy')
        plt.plot(wavelength_m[mid_idxs]*1e9,ring_spec[mid_idxs],marker='x',color='magenta')
        plt.xlabel("Wavelength (nm)",fontsize=13)
        plt.ylabel("Power (dBm)",fontsize=13)
        plt.show()
        fig.savefig(os.path.join(save_dir, 'Background_poly.png'), dpi=300)
        
        
    return flattened_spec

def Get_FSRs(Res_Wvls):
    '''
    Function to calculate FSR in nm from an array of resonances, interpolates each FSR as the point halfway between resonances
    '''
    # FSR values ad diff of wvls
    FSRs_m = np.diff(Res_Wvls) * 1e-9
    # wvls for each FSR point
    Midpt_Wvls_m = 0.5e-9 * (Res_Wvls[:-1] + Res_Wvls[1:])
    # interpolate FSR at resonance points
    FSR_interp_m = np.interp(Res_Wvls*1e-9, Midpt_Wvls_m, FSRs_m) if len(FSRs_m) > 0 else np.zeros_like(Res_Wvls)
    return FSRs_m, Midpt_Wvls_m, FSR_interp_m

def Get_GVD(Res_Wvls,ng):
    # Take second derivative of ng vs λ
    ng_spline = UnivariateSpline(Res_Wvls, ng, k=3, s=0)
    d2ng_dlambda2 = ng_spline.derivative(n=2)(Res_Wvls)

    # GVD in units of s^2/m
    c = 299792458  # speed of light in m/s
    beta2 = - (Res_Wvls ** 3) / (2 * np.pi * c**2) * d2ng_dlambda2
    
    return beta2

def Get_Finesse(FSR_interp_m,FWHMs):
    Finesse = FSR_interp_m*1e9 / np.array(FWHMs)
    return Finesse