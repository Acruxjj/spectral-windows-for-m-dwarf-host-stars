# Spectral Window Dependence in Stellar Parameter Inference of M Dwarfs

Accurate characterization of M dwarf host stars is essential for reliable exoplanet science, as uncertainties in effective temperature (Teff), surface gravity (log g), and metallicity ([Fe/H]) propagate directly into planetary radius estimates, incident flux, habitable zone boundaries, and bulk composition. Spectral regions encode astrophysical information unevenly and exhibit varying sensitivity to stellar parameters and to systematic discrepancies between synthetic and observed spectra (the synthetic gap), whose profile depends on the spectral synthesis code adopted. We investigate dependence of stellar parameter inference on spectral window selection in M dwarfs using high-resolution synthetic spectra generated with Korg (1D LTE), training 1D convolutional neural networks to estimate Teff, log g, and [Fe/H] from multiple spectral regions, including the 8800-8835 Å window previously identified as optimal in PHOENIX-based studies. A comparison across windows under progressively more realistic conditions allows us to assess whether windows identified as optimal in one synthetic domain generalize to others, and to quantify how window-dependent parameter uncertainties propagate into exoplanet characterization, including habitable zone estimates and planetary radius inference.

## Setup

```bash
pip install -r requirements.txt
```

Korg runs on Julia, so the first time you use it install the Julia package:

```python
from src.datagen import install_korg
install_korg()  # slow, only needs to run once
```

## References

* Bello-García, A. et al. (2023).  *A&A* , 673, A105.
* Bonfils, X. et al. (2007).  *A&A* , 474, 293.
* Fischer, D. A. & Valenti, J. (2005).  *ApJ* , 622, 1102.
* Passegger, V. M. et al. (2020).  *A&A* , 642, A22.
* Schweitzer, A. et al. (2019).  *A&A* , 625, A68.
* Wheeler, A. et al. (2023).  *AJ* , 165, 68.
