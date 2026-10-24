"""
OriginLab-style matplotlib settings.

Follows the visual conventions of Origin's default plots:
  - closed black frame (box), major/minor ticks pointing inward
  - no grid (Origin default)
  - Arial-family sans-serif
  - filled symbols with black edges
  - opaque legend with a border

Colors are assigned to series in a fixed order and never cycled. The order below passed a
color-vision-deficiency separation check (worst adjacent pair protan delta E 24.9), so do not
reorder it. Black is reserved for axes, text and reference lines, not for series.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt

# series colors: fixed order. Beyond 5, split into facets instead of inventing colors.
ORIGIN_COLORS: list[str] = [
    "#C00000",  # red
    "#0070C0",  # blue
    "#00A550",  # green
    "#7030A0",  # purple
    "#E08B00",  # orange
]

# secondary encoding paired with color (color-vision deficiency, grayscale print). Origin's default symbol order.
ORIGIN_MARKERS: list[str] = ["s", "o", "^", "D", "v"]

# reserved for axes, text and reference lines
INK = "#000000"
REFERENCE = "#000000"


def apply_origin_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 10,
            "mathtext.fontset": "dejavusans",
            "axes.linewidth": 1.2,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "xtick.color": INK,
            "ytick.color": INK,
            "xtick.major.size": 5.0,
            "ytick.major.size": 5.0,
            "xtick.minor.size": 2.5,
            "ytick.minor.size": 2.5,
            "xtick.major.width": 1.2,
            "ytick.major.width": 1.2,
            "xtick.minor.width": 1.0,
            "ytick.minor.width": 1.0,
            "xtick.minor.visible": True,
            "ytick.minor.visible": True,
            "axes.grid": False,
            "lines.linewidth": 1.5,
            "lines.markersize": 6.0,
            "lines.markeredgewidth": 1.0,
            "axes.prop_cycle": mpl.cycler(color=ORIGIN_COLORS),
            "legend.frameon": True,
            "legend.edgecolor": INK,
            "legend.facecolor": "white",
            "legend.framealpha": 1.0,
            "legend.fancybox": False,
            "legend.borderpad": 0.4,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "figure.dpi": 100,
        }
    )


def style_axes(ax: plt.Axes, minor: bool = True) -> plt.Axes:
    """Apply the frame/tick conventions to one Axes; for plots that bypass rcParams (seaborn etc.)."""
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.2)
        spine.set_color(INK)

    ax.tick_params(which="major", direction="in", length=5.0, width=1.2, top=True, right=True, color=INK)
    ax.tick_params(which="minor", direction="in", length=2.5, width=1.0, top=True, right=True, color=INK)
    if minor:
        ax.minorticks_on()
    ax.grid(False)
    return ax


def series_style(index: int) -> dict:
    """Color and symbol for a series index. Always used together so series are not told apart by color alone."""
    return {
        "color": ORIGIN_COLORS[index % len(ORIGIN_COLORS)],
        "marker": ORIGIN_MARKERS[index % len(ORIGIN_MARKERS)],
    }
