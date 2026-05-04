import math
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

# Limits of the SDSS MARCS grid as used by Korg.interpolate_marcs.
# Source: Korg.jl documentation, Korg.Fit.ews_to_stellar_parameters
# default parameter_ranges, and Wheeler et al. (2024), arXiv:2310.19823.
_MARCS_TEFF_MIN = 2800.0
_MARCS_TEFF_MAX = 8000.0
_MARCS_LOGG_MIN = -0.5
_MARCS_LOGG_MAX = 5.5
_MARCS_FEH_MIN  = -2.5
_MARCS_FEH_MAX  =  1.0

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
    wavelength_window_A: tuple[float, float] | None = None,
    R: float | None = None,
    vsini: float | None = None,
    vmic: float | None = None,
    max_draw_attempts: int = 500,
    show_progress: bool = True,
    synthesize: bool = True,
    korg=None,
) -> list[dict]:
    """
    Draw `num_samples` random (Teff, logg, [Fe/H]) and optionally synthesize
    each spectrum.

    Radius is derived from mass via the Schweitzer et al. (2019) empirical
    M-R relation (A&A 625, A68, eq. 7) so that (M, R) pairs stay on the
    observed M-dwarf sequence rather than filling an unphysical rectangle.

    Each record carries the input parameters and, if synthesize=True, the Korg
    output arrays (wavelength, normalized flux, continuum). Samples that raise
    inside Korg are silently re-drawn up to `max_draw_attempts` times.

    Parameters
    ----------
    num_samples : int
        Number of stellar parameter sets (and spectra, if synthesize=True)
        to generate.
    teff_range_K : tuple[float, float]
        Uniform sampling range for effective temperature [K].
    feh_range_dex : tuple[float, float]
        Clip range for [Fe/H] drawn from the Bonfils et al. (2007) Gaussian
        (mu=-0.05, sigma=0.20 dex).
    mass_range_msun : tuple[float, float]
        Uniform sampling range for stellar mass [M_sun].
    wavelength_window_A : tuple[float, float] | None
        Spectral window (lambda_min, lambda_max) in Ångströms passed to Korg.
        Required when synthesize=True, ignored when synthesize=False.
    R : float | None
        Resolving power passed to Korg (optional).
    vsini : float | None
        Projected rotational velocity [km/s] passed to Korg (optional).
    vmic : float | None
        Microturbulent velocity [km/s] passed to Korg (optional).
    max_draw_attempts : int
        Maximum re-draw attempts per sample before raising RuntimeError.
    show_progress : bool
        Whether to display a tqdm progress bar.
    synthesize : bool
        If True (default), synthesizes spectra via Korg and returns records
        with wavelength_A, flux and continuum keys. If False, returns only
        the physical stellar parameters without invoking Korg. In both modes,
        only stars within the MARCS grid limits are accepted.
    korg : module | None
        Korg instance. Loaded automatically if None and synthesize=True.

    Returns
    -------
    list[dict]
        If synthesize=True: each dict contains Teff_K, logg_cgs, FeH_dex,
        Mass_Msun, Radius_Rsun, wavelength_A, flux, continuum.
        If synthesize=False: each dict contains only Teff_K, logg_cgs,
        FeH_dex, Mass_Msun, Radius_Rsun.
    """
    if synthesize:
        if wavelength_window_A is None:
            raise ValueError(
                "wavelength_window_A is required when synthesize=True."
            )
        if korg is None:
            korg = _load_korg()

    # kwargs is built once here because R, vsini, and vmic are call-level constants;
    # move inside the loop if any of these become per-sample drawn parameters.
    kwargs = {}
    if R is not None:
        kwargs["R"] = R
    if vsini is not None:
        kwargs["vsini"] = vsini
    if vmic is not None:
        kwargs["vmic"] = vmic

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

            record = {
                "Teff_K": teff_K,
                "logg_cgs": logg_cgs,
                "FeH_dex": feh_dex,
                "Mass_Msun": mass_msun,
                "Radius_Rsun": radius_rsun,
            }

            if not synthesize:
                # Limits of the SDSS MARCS grid as used by Korg.interpolate_marcs.
                # Source: Korg.jl documentation, Korg.Fit.ews_to_stellar_parameters
                # default parameter_ranges, and Wheeler et al. (2024), arXiv:2310.19823.
                # The cool dwarf regime (Teff <= 4000 K, logg >= 3.5) uses a dedicated
                # cubic spline scheme; below 2800 K no grid points exist and synthesis fails.
                # Stars outside the grid are rejected at sampling time to ensure all
                # generated parameters are synthesizable.
                if not (
                    _MARCS_TEFF_MIN <= teff_K  <= _MARCS_TEFF_MAX
                    and _MARCS_LOGG_MIN <= logg_cgs <= _MARCS_LOGG_MAX
                    and _MARCS_FEH_MIN  <= feh_dex  <= _MARCS_FEH_MAX
                ):
                    continue
                records.append(record)
                if pbar:
                    pbar.update(1)
                break

            try:
                wavelengths_A, flux_norm, cont_norm = korg.synth(
                    Teff=teff_K,
                    logg=logg_cgs,
                    M_H=feh_dex,
                    wavelengths=tuple(wavelength_window_A),
                    **kwargs,
                )
                records.append(
                    {
                        **record,
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


def synthesize_spectra_from_stars(
    df_stars: pd.DataFrame,
    wavelength_window_A: tuple[float, float],
    R: float | None = None,
    vsini: float | None = None,
    vmic: float | None = None,
    show_progress: bool = True,
    korg=None,
) -> list[dict]:
    """
    Synthesize spectra for each star in df_stars over a single wavelength window.

    Parameters
    ----------
    df_stars : pd.DataFrame
        DataFrame produced by generate_synthetic_spectra_dataset(..., synthesize=False).
        Must contain columns: Teff_K, logg_cgs, FeH_dex. Optionally, vsini and
        vmic columns can be present and will be used per-star, taking priority
        over the function-level vsini and vmic parameters.
    wavelength_window_A : tuple[float, float]
        Spectral window (lambda_min, lambda_max) in Ångströms passed to Korg.
    R : float | None
        Resolving power passed to Korg (optional). Instrumental constant,
        does not vary per star.
    vsini : float | None
        Projected rotational velocity [km/s]. Used as fallback if df_stars
        does not contain a vsini column.
    vmic : float | None
        Microturbulent velocity [km/s]. Used as fallback if df_stars does
        not contain a vmic column.
    show_progress : bool
        Whether to display a tqdm progress bar.
    korg : module | None
        Korg instance. Loaded automatically if None.

    Returns
    -------
    list[dict]
        Each dict contains all columns from df_stars for that star plus
        wavelength_A, flux and continuum arrays from Korg. Stars that fall
        outside the MARCS grid are silently skipped — check len(result) vs
        len(df_stars) to audit discarded samples.
    """
    if korg is None:
        korg = _load_korg()

    # R is built once here because it is an instrumental constant;
    # vsini and vmic are resolved per-star inside the loop.
    kwargs = {}
    if R is not None:
        kwargs["R"] = R

    records: list[dict] = []
    iterable = (
        tqdm(df_stars.iterrows(), total=len(df_stars), desc="Synthesizing spectra...", unit="sample")
        if show_progress
        else df_stars.iterrows()
    )

    for _, star in iterable:
        # vsini and vmic are stellar properties — resolved per star.
        # Priority: df column (per-star) > function parameter (fallback) > omit.
        star_kwargs = {**kwargs}
        if "vsini" in star.index and pd.notna(star["vsini"]):
            star_kwargs["vsini"] = star["vsini"]
        elif vsini is not None:
            star_kwargs["vsini"] = vsini

        if "vmic" in star.index and pd.notna(star["vmic"]):
            star_kwargs["vmic"] = star["vmic"]
        elif vmic is not None:
            star_kwargs["vmic"] = vmic


        # Limits of the SDSS MARCS grid as used by Korg.interpolate_marcs.
        # Source: Korg.jl documentation, Korg.Fit.ews_to_stellar_parameters
        # default parameter_ranges, and Wheeler et al. (2024), arXiv:2310.19823.
        # The cool dwarf regime (Teff <= 4000 K, logg >= 3.5) uses a dedicated
        # cubic spline scheme; below 2800 K no grid points exist and synthesis fails.
        # Stars outside the grid are silently skipped — check len(records) vs len(df_stars).

        try:
            wavelengths_A, flux_norm, cont_norm = korg.synth(
                Teff=star["Teff_K"],
                logg=star["logg_cgs"],
                M_H=star["FeH_dex"],
                wavelengths=tuple(wavelength_window_A),
                **star_kwargs,
            )
            records.append(
                {
                    **star.to_dict(),
                    "wavelength_A": wavelengths_A,
                    "flux": flux_norm,
                    "continuum": cont_norm,
                }
            )
        except Exception:
            continue

    return records