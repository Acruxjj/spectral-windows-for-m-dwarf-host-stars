from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

_STYLE = {
    "xtick.top": True,
    "ytick.right": True,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "font.size": 14,
    "font.family": "sans-serif",
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "legend.fontsize": 12,
    "legend.frameon": False,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "xtick.major.size": 6,
    "ytick.major.size": 6,
    "xtick.minor.size": 2,
    "ytick.minor.size": 2,
}

plt.rcParams.update(_STYLE)


def plot_single_spectrum(df_spectra: pd.DataFrame, row_index: int):
    """
    Plot a single continuum-normalized spectrum from `df_spectra`.
    """
    row = df_spectra.iloc[row_index]
    wavelength_A = row["wavelength_A"]
    flux = row["flux"]

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(wavelength_A, flux)
    ax.set_xlabel(r"Wavelength $\lambda$ [$\AA$]")
    ax.set_ylabel("continuum-normalized flux")

    title = (
        f"Teff={row['Teff_K']:.0f} K  "
        f"logg={row['logg_cgs']:.2f}  "
        f"[Fe/H]={row['FeH_dex']:.2f}"
    )

    ax.set_title(title, loc="left", fontweight="bold")
    return fig, ax


def plot_histogram(
    df: pd.DataFrame,
    column: str,
    value_range: tuple[float, float],
    step: float,
    xlabel: str | None = None,
    ylabel: str = "Count",
    figsize: tuple[float, float] = (10, 4),
    color: str = "tab:blue",
):
    """
    Histogram of `df[column]` with bins defined by `value_range` and `step`.
    """
    bins = np.arange(value_range[0], value_range[1] + step, step)

    fig, ax = plt.subplots(figsize=figsize)

    ax.hist(df[column], bins=bins, edgecolor="k", color=color)

    ax.set_xlabel(xlabel if xlabel else column)
    ax.set_ylabel(ylabel)
    ax.set_xticks(bins)

    return fig, ax


def plot_spectra_colored_by_params(
    df: pd.DataFrame,
    params: Sequence[tuple[str, str, str]] = (
        ("Teff_K", "Teff [K]", "plasma"),
        ("logg_cgs", "log g [cgs]", "viridis"),
        ("FeH_dex", "[Fe/H] [dex]", "cividis"),
    ),
    figsize: tuple[float, float] = (12, 10),
):
    """
    Stack of spectra panels, each colored by a different stellar parameter.
    """
    fig, axes = plt.subplots(len(params), 1, figsize=figsize, sharex=True)

    for ax, (col, label, cmap_name) in zip(axes, params):
        norm = Normalize(vmin=df[col].min(), vmax=df[col].max())
        cmap = plt.get_cmap(cmap_name)

        for _, row in df.iterrows():
            ax.plot(
                np.asarray(row["wavelength_A"]),
                np.asarray(row["flux"]),
                color=cmap(norm(row[col])),
            )

        sm = ScalarMappable(norm=norm, cmap=cmap)

        fig.colorbar(sm, ax=ax, label=label, pad=0)
        ax.set_ylabel("Continuum-normalized flux")

    axes[-1].set_xlabel(r"Wavelength $\lambda$ [$\AA$]")

    fig.tight_layout()
    return fig, axes
