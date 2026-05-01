import math
import numpy as np
from tqdm.auto import tqdm


def install_korg() -> None:
    """
    One-time install of the Korg Julia package into the juliapkg env.

    Run this once after cloning the repo (and again whenever you blow
    away the Julia env). It is slow, expect several minutes the first
    time because juliapkg also downloads Julia itself if it is missing.

    https://ajwheeler.github.io/Korg.jl/stable/install/#Using-Korg-from-Python
    """
    import juliapkg

    # juliacall 0.9.27 currently resolves cleanly with Julia 1.10.x here.
    # Letting juliapkg pick the newest Julia can select Julia 1.12 and produce
    # an OpenSSL_jll compat conflict before Korg is even installed.
    juliapkg.require_julia("~1.10.0")

    # Resolve before importing juliacall. Importing juliacall starts Julia, so
    # Korg has to be present in the managed environment first.
    juliapkg.add("Korg", "acafc109-a718-429c-b0e5-afd7f8c7ae46")
    juliapkg.resolve()

    from juliacall import Main as jl

    jl.seval("using Korg")


def compute_logg_from_mass_radius(mass_msun: float, radius_rsun: float) -> float:
    """
    Surface gravity log g (cgs) from mass and radius in solar units.

    log10(g) = log10(M/Msun) - 2*log10(R/Rsun) + 4.437
    Ref: https://adsabs.harvard.edu/full/2005MSAIS...8..130S
    """
    return math.log10(mass_msun) - 2.0 * math.log10(radius_rsun) + 4.437


def radius_from_mass(mass_msun: float) -> float:
    """
    Radius from mass using Schweitzer et al. (2019), A&A 625, A68, eq. 7.

    R/Rsun = 0.0282 + 0.935 · (M/Msun),  rms ≈ 0.02 Rsun
    """
    r = 0.0282 + 0.935 * mass_msun + float(np.random.normal(0.0, 0.02))
    return max(r, 0.05)


def _load_korg():
    # Imported lazily so callers that don't generate spectra
    # don't pay the Julia startup cost.
    from juliacall import Main as jl

    jl.seval("using Korg")
    return jl.Korg


def generate_synthetic_spectra_dataset(
    num_samples: int,
    teff_range_K: tuple[float, float],
    feh_range_dex: tuple[float, float],
    mass_range_msun: tuple[float, float],
    wavelength_window_A: tuple[float, float],
    R: float | None = None,
    vsini: float | None = None,
    vmic: float | None = None,
    max_draw_attempts: int = 500,
    show_progress: bool = True,
    korg=None,
) -> list[dict]:
    """
    Draw `num_samples` random (Teff, logg, [Fe/H]) and synthesize each spectrum.

    Radius is derived from mass via the Schweitzer et al. (2019) empirical
    M-R relation (A&A 625, A68, eq. 7) so that (M, R) pairs stay on the
    observed M-dwarf sequence rather than filling an unphysical rectangle.

    Each record carries the input parameters and the Korg output arrays
    (wavelength, normalized flux, continuum). Samples that raise inside
    Korg are silently re-drawn up to `max_draw_attempts` times.
    """
    if korg is None:
        korg = _load_korg()

    records: list[dict] = []
    pbar = (
        tqdm(total=num_samples, desc="Synthesizing spectra...", unit="sample")
        if show_progress
        else None
    )

    while len(records) < num_samples:
        attempts = 0
        while attempts < max_draw_attempts:
            attempts += 1

            teff_K = float(np.random.uniform(*teff_range_K))
            # [Fe/H] drawn from the solar-neighbourhood M-dwarf distribution
            # (Gaussian, not uniform) then clipped to feh_range_dex.
            # Bonfils et al. (2007), A&A 474, 293: mu ~ -0.05, sigma ~ 0.20 dex
            feh_dex = float(np.clip(np.random.normal(-0.05, 0.20), *feh_range_dex))
            mass_msun = float(np.random.uniform(*mass_range_msun))
            radius_rsun = radius_from_mass(mass_msun)
            logg_cgs = compute_logg_from_mass_radius(mass_msun, radius_rsun)

            try:
                kwargs = {}
                if R is not None:
                    kwargs["R"] = R
                if vsini is not None:
                    kwargs["vsini"] = vsini
                if vmic is not None:
                    kwargs["vmic"] = vmic

                wavelengths_A, flux_norm, cont_norm = korg.synth(
                    Teff=teff_K,
                    logg=logg_cgs,
                    M_H=feh_dex,
                    wavelengths=tuple(wavelength_window_A),
                    **kwargs,
                )
                records.append(
                    {
                        "Teff_K": teff_K,
                        "logg_cgs": logg_cgs,
                        "FeH_dex": feh_dex,
                        "Mass_Msun": mass_msun,
                        "Radius_Rsun": radius_rsun,
                        "wavelength_A": wavelengths_A,
                        "flux": flux_norm,
                        "continuum": cont_norm,
                    }
                )
                if pbar:
                    pbar.update(1)
                break

            except Exception:
                continue

        if attempts >= max_draw_attempts:
            if pbar:
                pbar.close()
            raise RuntimeError("Could not draw a valid point")

    if pbar:
        pbar.close()

    return records
