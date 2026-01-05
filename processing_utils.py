# processing_utils.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def Calculate_Q(lam0, FWHM):
    '''
    Calculate the Q-factor from FWHM and central wavelength
        FWHM: Full width half-max
        lam0: central wavelength
    '''
    FWHM = abs(FWHM)
    Q = lam0 / FWHM if FWHM != 0 else np.inf
    return Q

def Lorentzian(x, A, x0, FWHM, offset):
    '''
    Lorentzian function definition
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
    print(f"alpha = {round(-10*np.log10(alpha_rt**2),3)} dB")
    
    alpha_dB_m = -10*np.log10(alpha_rt**2)/(2*np.pi*r_ring)
    alpha_dB_cm = alpha_dB_m/100
    print(f"{round(alpha_dB_cm,3)} dB/cm")
    return alpha_dB_cm

def Calculate_Coupler_Loss(t):
    '''
    Calculate coupler loss from self coupling coefficient
        t: self coupling coefficient
    '''
    t_dB = -10*np.log10(t**2)
    print(f"t = {round(t_dB,3)} dB")
    return t_dB

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

def Group_Index(FSR,ring_rad,fsr_centers_m):
    '''
    calculate group index from FSR, ring radius and wvl
    ng = λ² / (L x FSR)
        FSR: Free spectral range
        ring_rad: ring radius
        fsr_centers_m: wvl points midway between resonances
    '''
    ring_length = 2*np.pi*ring_rad
    ng = (fsr_centers_m ** 2) / (FSR * ring_length)
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

def fit_lorentzian_wvl(wavelength, spectrum, peak_index, window_nm=0.4):
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