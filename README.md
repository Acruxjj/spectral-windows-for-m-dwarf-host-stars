# Identification of Spectral Windows for Stellar Parameter Inference in M Dwarfs and Their Implications for Exoplanet Characterization

This project investigates how the choice of spectral windows affects the inference of stellar parameters (Teff, log g, and [Fe/H]) in M dwarfs using synthetic spectra generated with Korg and 1D convolutional neural networks. Building on previous results obtained in the 8800–8830 Å region, the current work explores multiple spectral windows and increasing levels of physical realism to identify regions that provide more robust parameter recovery. The goal is to assess how spectral window selection and synthetic-domain limitations contribute to the synthetic gap and how these effects propagate into exoplanet characterization.

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

These baseline tests were performed using simplified synthetic spectra in the 8800–8830 Å spectral window, focusing on the simultaneous regression of:

* Effective Temperature (Teff)
* Surface Gravity (log g)
* Metallicity ([Fe/H])

The spectra were generated using Korg under simplified assumptions, without instrumental degradation to the CARMENES resolution, without projected rotational velocity (v sin i), and without detailed chemical abundance variations beyond the adopted metallicity grid.

This spectral window was selected based on previous literature indicating that the 8800–8835 Å region provides the smallest validation error among the tested spectral ranges for M-dwarf parameter inference.

The network converged in approximately 27 epochs without evidence of overfitting, achieving (R² = 0.998) for log g and (R² = 0.999) for [Fe/H], with MAE < 0.02 for both parameters and an MAE of approximately 6 K for Teff.

These results demonstrate that the CNN is capable of recovering stellar parameters with strong predictive performance in a controlled synthetic domain, establishing the methodological baseline for the project.

The `notebooks/spectral_windows_analysis/` directory contains the current stage of the project, focused on comparing multiple spectral windows, increasing the physical realism of synthetic spectra, and investigating the impact of stellar parameter uncertainties on exoplanet characterization.
