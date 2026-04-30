# Identification of Spectral Windows for Stellar Parameter Inference in M Dwarfs and Their Implications for Exoplanet Characterization

This project studies how spectral window selection affects the inference of stellar parameters (Teff, log g, and [Fe/H]) in M dwarfs using synthetic spectra generated with Korg, a 1D LTE spectral synthesis code, and one-dimensional convolutional neural networks (1D CNNs). Building on previous results in the 8800–8830 Å region, the current work extends the analysis to multiple spectral windows and progressively more realistic synthetic conditions to evaluate parameter sensitivity across different wavelength ranges. The goal is to assess how spectral window choice and limitations of synthetic models contribute to the synthetic gap and how these effects propagate into exoplanet characterization.

---

## Project Structure

data/ → synthetic spectra and processed datasets

figures/ → plots and scientific visualization

notebooks/ → Google Colab notebooks

results/ → model outputs and metrics

src/ → modular Python scripts

---

## Project Status

The `notebooks/baseline/` directory contains the initial proof-of-concept experiments used to validate the 1D CNN architecture for stellar parameter inference.

Baseline tests were performed using simplified synthetic spectra in the 8800–8830 Å region, focusing on the simultaneous regression of effective temperature (Teff), surface gravity (log g), and metallicity ([Fe/H]).

The spectra were generated with Korg, a 1D LTE spectral synthesis code, under simplified assumptions, without instrumental degradation to the CARMENES resolution, without projected rotational velocity (v sin i), and without detailed chemical abundance variations beyond the adopted metallicity grid.

The 8800–8835 Å region was selected based on previous literature reporting the smallest validation error among tested spectral ranges for M-dwarf parameter inference.

The network converged in approximately 27 epochs without evidence of overfitting, achieving R² = 0.998 for log g and R² = 0.999 for [Fe/H], with MAE < 0.02 for both parameters and approximately 6 K for Teff.

These results establish a controlled synthetic baseline for the project, demonstrating that the architecture can recover stellar parameters with high predictive performance under simplified assumptions.

The `notebooks/spectral_windows_analysis/` directory contains the current stage of the project, focused on comparing multiple spectral windows, increasing the physical realism of synthetic spectra, and investigating the impact of stellar parameter uncertainties on exoplanet characterization.
