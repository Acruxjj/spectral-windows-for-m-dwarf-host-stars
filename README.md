# Spectral Window Dependence in Stellar Parameter Inference of M Dwarfs

Accurate characterization of M dwarfs is essential for understanding stellar and planetary system evolution, yet atmospheric parameter inference remains sensitive to spectral window selection, a limitation further compounded by systematic discrepancies between synthetic and observed spectra (the synthetic gap). We trained 1D Convolutional Neural Networks on 5,000 synthetic spectra computed with Korg at CARMENES resolution using physically coupled stellar parameters, evaluating four spectral windows: a positive control (8800-8835 Å), Ca II IRT (8640-8710 Å), TiO (8160-8225 Å), and K I + TiO (7640-7725 Å). TiO bands achieved the best performance for log g (R² = 0.962; MAE = 0.024 dex) and [Fe/H] (R² = 0.982; MAE = 0.019 dex), while the control window yielded the lowest Teff error (MAE = 20.2 K). Robustness tests against rotational broadening (v sin i = 0-120 km s⁻¹) and microturbulence (vmic = 0-3 km s⁻¹) across seven subtypes (M0V-M6V) and three metallicities ([Fe/H] = -0.5, 0.0, +0.2) revealed an accuracy-robustness inversion: the control window is most stable at low rotational velocities (v sin i ≲ 20 km s⁻¹), typical of field M dwarfs, showing that the highest-performing window under ideal conditions is not necessarily the most reliable under physical perturbations. We also show that these parameter residuals propagate into exoplanet observables, with window selection producing differences of up to 3.4% in the habitable-zone location for a 3300 K M dwarf, 4.1% in transit-derived planetary radius, and ~20% in the inferred giant planet occurrence rate. The control window minimized habitable zone propagation; TiO bands produced the smallest errors in planetary radius and giant planet frequency; Ca II IRT and K I + TiO showed the largest degradations depending on the observable. These results establish a quantitative baseline derived entirely from synthetic spectra; the next step is to confront this baseline against real CARMENES observations to quantify the synthetic gap and assess how it propagates into the same exoplanet observables.


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
