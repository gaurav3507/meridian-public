"""Shared publication-quality matplotlib style for all paper figures.

Import at the top of every figure script:  import paper_style
Sets 300-dpi output, small journal font sizes, editable (Type-42) fonts in
PDF/PS, and tight bounding boxes. Call paper_style.apply() again AFTER any
seaborn set_theme/set (which resets font rcParams) to re-assert these sizes.
"""
import matplotlib as mpl

mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['figure.dpi'] = 150
mpl.rcParams['font.size'] = 8
mpl.rcParams['axes.titlesize'] = 8
mpl.rcParams['axes.labelsize'] = 8
mpl.rcParams['xtick.labelsize'] = 7
mpl.rcParams['ytick.labelsize'] = 7
mpl.rcParams['legend.fontsize'] = 7
mpl.rcParams['pdf.fonttype'] = 42   # editable text, not outlines
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['savefig.bbox'] = 'tight'

# physical figure widths (inches)
SINGLE_COL = 3.5
DOUBLE_COL = 7.2


def apply():
    """Re-assert the paper rcParams (call after seaborn set_theme/set)."""
    mpl.rcParams['savefig.dpi'] = 300
    mpl.rcParams['figure.dpi'] = 150
    mpl.rcParams['font.size'] = 8
    mpl.rcParams['axes.titlesize'] = 8
    mpl.rcParams['axes.labelsize'] = 8
    mpl.rcParams['xtick.labelsize'] = 7
    mpl.rcParams['ytick.labelsize'] = 7
    mpl.rcParams['legend.fontsize'] = 7
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42
    mpl.rcParams['savefig.bbox'] = 'tight'
