import argparse
import json

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from iblatlas.atlas import AllenAtlas

from iblnm.config import SESSIONS_FPATH, QCPHOTOMETRY_FPATH, TRAJECTORIES_FPATH
from iblnm.io import _get_default_connection, get_fiber_coordinates


# Parse command line arguments
parser = argparse.ArgumentParser(
    description='Get a fiber insertion trajectory and plot on a brain slice.'
    )
parser.add_argument('--subject', '-s')
parser.add_argument('--fiber', '-f')
parser.add_argument('--qc_metric', '-q')
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

print(f"Getting trajectory for {args.subject}...")
coords = get_fiber_coordinates(args.subject, all_trajectories, one)

# FIXME: QC val is curently aggregate over all recordings x fibers for the subject
# Once probe_names in Alyx can be mapped on to photometry columns we need to fix
if args.qc_metric and QCPHOTOMETRY_FPATH.exists() and SESSIONS_FPATH.exists():
    print(f"Getting average {args.qc_metric} QC value for {args.subject}...")
    df_qc = pd.read_parquet(QCPHOTOMETRY_FPATH)
    df_sessions = pd.read_parquet(SESSIONS_FPATH)
    eids = df_sessions[df_sessions['subject'] == args.subject]['eid']
    if len(eids) > 0:
        qc_val = df_qc[df_qc['eid'].isin(eids)][args.qc_metric].mean()
        clevels = df_qc[args.qc_metric].min(), df_qc[args.qc_metric].max()
else:
    print(f"No photometry QC values for subject {args.subject}.")
    qc_val = 1.
    clevels = None

# Assign point color based on QC value
cmap = plt.cm.Reds
norm = plt.Normalize(vmin=clevels[0], vmax=clevels[1])
facecolor = cmap(norm(qc_val))

ba = AllenAtlas()
# x, y axes for each slice view (indices into ML=0, AP=1, DV=2 coord array)
slice_xy = {'coronal': (0, 2), 'sagittal': (1, 2), 'horizontal': (0, 1)}
slice_fn = {'coronal': ba.plot_cslice, 'sagittal': ba.plot_sslice, 'horizontal': ba.plot_hslice}
slice_coord_idx = {'coronal': 1, 'sagittal': 0, 'horizontal': 2}

views = ['sagittal', 'coronal', 'horizontal']
fig, axs = plt.subplots(len(coords), 3, figsize=(10, 10))
if axs.ndim == 1:
    axs = axs[np.newaxis, :]
for (fiber, coord), ax_row in zip(coords.items(), axs):
    for view, ax in zip(views, ax_row):
        ax.set_title(fiber)
        slice_fn[view](coord[slice_coord_idx[view]] / 1e6, volume='boundary', ax=ax)
        xi, yi = slice_xy[view]
        ax.scatter(
            coord[xi],
            coord[yi],
            s=100,
            facecolors=facecolor,
            edgecolors=cmap(1.),
            linewidths=2,
            zorder=10
        )

