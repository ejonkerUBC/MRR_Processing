import numpy as np
import pandas as pd
import os
from pathlib import Path
import utils
import scripts

# ==== USER PARAMETERS ====
CONFIG = {
    "lower_wl_limit": 1545e-9,     # m
    "upper_wl_limit": 1565e-9,     # m
    "res_ER_thrshld": 1.0,      # dBm (ER required to identify a resonance)
    "Show_Resonance_Plots": False, # show fitting plot for each resonance 
    "Show_Other_Plots": True, # show the other plots (spectrum, other ring params ect..)
    "Flatten_Spectrum": True,
    "Ring_Rad": 80e-6,
}
# ==========================

# === Main Processing ===
script_dir = Path(__file__).resolve().parent
csv_path = script_dir/"Example Data"/"ring_resonance_raw_spectrum.csv"
figures_dir = script_dir/"Figures"

ring_length = 2*np.pi*CONFIG["Ring_Rad"]

ring_wl, ring_frq, ring_spec = utils.load_data_scylla_csv(csv_path, spectrum_key='channel_1')

spectrum = utils.Find_Background_Poly(ring_wl, ring_spec, FSR_m=2e-9, DoPlot=True, save_dir=figures_dir) if CONFIG["Flatten_Spectrum"] else ring_spec

in_range_peaks = utils.Find_Resonance_Idxs(ring_wl, spectrum, save_dir=figures_dir, wvl_range=[CONFIG["lower_wl_limit"], CONFIG["upper_wl_limit"]], FSR_m=2e-9, DoPlot=True)

# === Analyze Resonances ===
Res_df = scripts.Analyze_Resonances(in_range_peaks,ring_wl*1e9,spectrum,DoPlot=CONFIG["Show_Resonance_Plots"],save_dir=figures_dir)
    
# Additional Calculations based on resonance measurements
if len(Res_df["wavelength"]) > 1:
    Res_Wvls = Res_df["wavelength"].values
    
   # === Calculate FSR ===
    FSRs_m, Midpt_Wvls_m, FSR_interp_m = utils.Get_FSRs(Res_Wvls)
    
    # === Calculate Group Index ===
    ng = utils.Group_Index(FSR_interp_m,CONFIG["Ring_Rad"],Res_Wvls*1e-9)
    
    # === Calculate Group Velocity ===
    vg = utils.Group_Velocity(ng)

    # === Calculate Group Velocity Dispersion (GVD) ===
    beta2 = utils.Get_GVD(Res_Wvls*1e-9,ng)
    
    # === Calculate Finesse ===
    Finesse = utils.Get_Finesse(FSR_interp_m,Res_df["FWHM"])
    
    # === Calculate roots (alpha and tau) ===
    root1_values, root2_values, kappa1_values, kappa2_values = utils.Calculate_Roots(Finesse,Res_df["ER_lin"])
    
    # If you know whether the ring/resonance is over or undercoupled we can calculate the coupling and roundtrip loss (alpha and tau)
    # -> undercoupled (tau > alpha) alpha is the smaller of {root1,root2}, and tau the larger
    # -> overcoupled (alpha > tau) tau is the smaller of {root1,root2}, and alpha the larger
    alpha = utils.Calculate_Prop_Loss(root1_values,CONFIG["Ring_Rad"])
    tau = utils.Calculate_Coupler_Loss(root2_values)

Measured_Details = {
    "Wvl":Res_Wvls,
    "FSR":FSR_interp_m,
    "ng":ng,
    "vg":vg,
    "GVD":beta2,
    "Finesse":Finesse,
    "root1":root1_values,
    "root2":root2_values,
    "kappa1":kappa1_values,
    "kappa2":kappa2_values,
}

# Plot all the calculated measurements in a single figure
scripts.plot_measurements(Res_df,Measured_Details,Midpt_Wvls_m,FSRs_m,save_dir=figures_dir)
scripts.roots_vs_wavelength(Measured_Details,Res_df["ER_lin"],save_dir=figures_dir)

# Some random other scripts 
scripts.ER_Finesse_vs_wvl(Measured_Details,Res_df)
scripts.Field_Enhancement_Factor(Measured_Details,Res_df["ER_lin"])
scripts.FSR_Freq(Midpt_Wvls_m,FSRs_m,Measured_Details)

Measured_Details = pd.DataFrame(Measured_Details)

csv_path = os.path.join(figures_dir, 'Measured_Data.csv')
Measured_Details.to_csv(csv_path, index=False)
print(f"Saved measured data to {csv_path}")



