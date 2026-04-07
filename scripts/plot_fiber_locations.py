"""
Plot Fiber Insertion Locations on Brain Slices

Retrieves chronic fiber insertion trajectories from Alyx and plots each
fiber's tip coordinates on coronal, sagittal, and horizontal brain atlas
slices. Fibers are grouped by brain_region (one row per region, one column
per view). The slice coordinate for each view is the median of the
corresponding axis across all fibers in that region.

Brain region is derived from df_sessions and only assigned when unambiguous
(single unique region per subject, ignoring hemisphere suffix).

Trajectories are cached in metadata/trajectories.json after the first
download; use --redownload to refresh the cache.

Usage
-----
    python plot_fiber_locations.py [--subject S1 S2 ...] [--redownload]

Examples
--------
    python plot_fiber_locations.py
    python plot_fiber_locations.py --subject SWC_054
    python plot_fiber_locations.py --redownload
"""
import argparse
import json

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from iblatlas.atlas import AllenAtlas

from iblnm.config import SESSIONS_FPATH, TARGETNMS_TO_ANALYZE, TRAJECTORIES_FPATH
from iblnm.io import _get_default_connection, get_fiber_coordinates

ba = AllenAtlas()
# x, y axes for each slice view (indices into ML=0, AP=1, DV=2 coord array)
slice_xy = {'coronal': (0, 2), 'sagittal': (1, 2), 'horizontal': (0, 1)}
slice_fn = {'coronal': ba.plot_cslice, 'sagittal': ba.plot_sslice, 'horizontal': ba.plot_hslice}
slice_coord_idx = {'coronal': 1, 'sagittal': 0, 'horizontal': 2}

# Parse command line arguments
parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter,
)
parser.add_argument('--subject', '-s', nargs='+', help='Alyx subject names (e.g. SWC_054)')
parser.add_argument('--fiber', '-f', metavar='FIBER',
                    help='Fiber name to plot (e.g. G0). If omitted, all fibers are plotted.')
# FIXME: use --fiber to restrict coords to the specified fiber only
parser.add_argument('--redownload', action='store_true',
                    help='Re-download trajectories from Alyx, ignoring cache')
args = parser.parse_args()

one = _get_default_connection()

if not args.redownload and TRAJECTORIES_FPATH.exists():
    print(f"Loading trajectories from {TRAJECTORIES_FPATH}...")
    with open(TRAJECTORIES_FPATH) as f:
        all_trajectories = json.load(f)
else:
    print("Downloading all trajectories from Alyx...")
    all_trajectories = list(one.alyx.rest('trajectories', 'list'))
    TRAJECTORIES_FPATH.parent.mkdir(parents=True, exist_ok=True)
    with open(TRAJECTORIES_FPATH, 'w') as f:
        json.dump(all_trajectories, f)
    print(f"Saved to {TRAJECTORIES_FPATH}")

print(f"Loading sessions from {SESSIONS_FPATH}")
df_sessions = pd.read_parquet(SESSIONS_FPATH)
print(f"Loaded {len(df_sessions)} sessions")

if not args.subject:
    # FIXME: put some session filtering here
    subjects = df_sessions['subject'].unique()
else:
    subjects = args.subject

coords = []
for subject in tqdm(subjects):
    coords.extend(get_fiber_coordinates(subject, all_trajectories, one))
df_coords = pd.DataFrame(coords)


# ---------------------------------------------------------------------------
# Merge brain_region from df_sessions
# ---------------------------------------------------------------------------
def _get_subject_region(subject, df_sessions):
    """Return the brain region for a subject if unambiguous, else None.

    Unambiguous means all brain_region entries across sessions for this
    subject resolve to the same base region after stripping hemisphere
    suffixes (-r, -l).
    """
    subject_sessions = df_sessions[df_sessions['subject'] == subject]
    regions = set()
    for br_list in subject_sessions['brain_region']:
        for r in br_list:
            base = r[:-2] if r.endswith(('-r', '-l')) else r
            regions.add(base)
    if len(regions) == 1:
        return regions.pop()
    return None


df_coords['brain_region'] = df_coords['subject'].map(
    lambda s: _get_subject_region(s, df_sessions))
n_before = len(df_coords)
df_coords = df_coords.dropna(subset=['brain_region'])
n_dropped = n_before - len(df_coords)
if n_dropped:
    print(f"Dropped {n_dropped} fibers with ambiguous brain_region")

targets_to_analyze = {t.split('-')[0] for t in TARGETNMS_TO_ANALYZE}
df_coords = df_coords[df_coords['brain_region'].isin(targets_to_analyze)]

# ---------------------------------------------------------------------------
# Plot: rows = brain_region, columns = view
# ---------------------------------------------------------------------------
views = ['coronal', 'sagittal', 'horizontal']
regions = sorted(df_coords['brain_region'].unique())

fig, axs = plt.subplots(len(regions), len(views),
                        figsize=(4 * len(views), 4 * len(regions)))
if axs.ndim == 1:
    axs = axs[np.newaxis, :]

for i, region in enumerate(regions):
    df_region = df_coords[df_coords['brain_region'] == region]
    all_coords = np.stack(df_region['coords'].values)  # (n_fibers, 3)
    median_coords = np.median(all_coords, axis=0)       # [ML, AP, DV]

    for j, view in enumerate(views):
        ax = axs[i, j]
        coord_idx = slice_coord_idx[view]
        slice_fn[view](median_coords[coord_idx] / 1e6, volume='boundary', ax=ax)

        xi, yi = slice_xy[view]
        ax.scatter(
            all_coords[:, xi],
            all_coords[:, yi],
            s=100,
            facecolors='steelblue',
            edgecolors='navy',
            linewidths=2,
            zorder=10,
        )

        if i == 0:
            ax.set_title(view.capitalize())
        if j == 0:
            ax.set_ylabel(region)

plt.tight_layout()
plt.show()
