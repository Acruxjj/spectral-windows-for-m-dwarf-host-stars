# Identification of Spectral Windows for Stellar Parameter Inference in M Dwarfs and Their Implications for Exoplanet Characterization

Spectral window analysis for stellar parameter inference in M dwarfs using synthetic spectra and 1D CNNs, focusing on the synthetic gap and the propagation of stellar parameter uncertainties into exoplanet characterization.

---

## Project Structure

data/ → synthetic spectra and processed datasets

figures/ → plots and scientific visualization

notebooks/ → Google Colab notebooks

results/ → model outputs and metrics

src/ → modular Python scripts

## Project Status

The `notebooks/baseline/` directory contains the initial proof-of-concept experiments used to validate the 1D CNN architecture for stellar parameter inference.

These baseline tests were performed using simplified synthetic spectra in the 8800–8830 Å spectral window, focusing on the simultaneous regression of:

* Effective Temperature (Teff)
* Surface Gravity (log g)
* Metallicity ([Fe/H])

The spectra were generated using Korg under simplified assumptions, without instrumental degradation to the CARMENES resolution, without projected rotational velocity ((v \sin i)), and without detailed chemical abundance variations beyond the adopted metallicity grid.

This spectral window was selected based on previous literature indicating that the 8800–8835 Å region provides the smallest validation error among the tested spectral ranges for M-dwarf parameter inference.

The network converged in approximately 27 epochs without evidence of overfitting, achieving (R^2 = 0.998) for log g and (R^2 = 0.999) for [Fe/H], with MAE < 0.02 for both parameters and an MAE of approximately 6 K for Teff.

These results demonstrate that the CNN is capable of recovering stellar parameters with strong predictive performance in a controlled synthetic domain, establishing the methodological baseline for the project.
