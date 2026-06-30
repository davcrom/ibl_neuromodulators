"""Single-session photometry encoding model.

Fits a kernel-based ridge encoding model for one recording: an FIR (or
raised-cosine) design built from behavioural events, parametric modulators, and
continuous movement regressors, fit with K-fold-tuned ridge, scored by
leave-one-regressor-out ΔR², and inspected with prediction/kernel/ΔR² plots.

The code-and-build step (events → coded modulators → design matrix) is the
``build_encoding_design`` function so the identical coding is testable without
ONE; the fit → evaluate → plot sequence is spelled out in the ``__main__`` body.

Usage:
    python scripts/encoding.py <eid> <brain_region>
"""
import argparse
from functools import partial
from typing import Callable

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')  # batch figure generation; never open interactive windows

from iblnm import task
from iblnm.config import (
    PROJECT_ROOT, SESSIONS_FPATH, SESSIONS_H5_DIR,
    ENCODING_DT, ENCODING_N_LAGS, ENCODING_ALPHAS, ENCODING_CV,
    ENCODING_TERMS, ENCODING_POSE_KEYPOINTS, FIGURE_DPI,
)
from iblnm.analysis import (
    make_time_grid, make_lags, lag_expand, interpolate_to_grid,
    build_event_blocks, build_continuous_block, build_design_matrix,
    deviation_code, keypoint_speed, fit_encoding_model,
)
from iblnm.data import PhotometrySessionGroup
from iblnm.io import _get_default_connection
from iblnm.vis import (
    plot_encoding_prediction, plot_encoding_kernels, plot_delta_r_squared,
)


# config modulator name -> trials column carrying its per-event value. `choice`
# enters as the chosen side relative to the recording hemisphere (choice_side).
MODULATOR_COLUMN = {'side': 'side', 'choice': 'choice_side', 'contrast': 'contrast'}


def _code_modulator(name: str, kind: str, trials: pd.DataFrame) -> np.ndarray:
    """Per-event modulator heights for one config modulator.

    Categorical modulators are deviation-coded ±0.5 (contra = +0.5, relative to
    the recording hemisphere); continuous modulators are mean-centered.
    """
    column = trials[MODULATOR_COLUMN[name]].to_numpy()
    if kind == 'categorical':
        return deviation_code(column, positive='contra')
    column = column.astype(float)
    return column - np.nanmean(column)


def build_encoding_design(
    trials: pd.DataFrame,
    target: pd.Series,
    hemisphere: str,
    expander: Callable[[np.ndarray], np.ndarray],
    continuous: dict[str, pd.Series] | None = None,
    terms: dict = ENCODING_TERMS,
) -> tuple[np.ndarray, dict[str, slice], pd.Series]:
    """Code predictors and assemble the encoding design matrix for one recording.

    Steps 4-7 of the analysis (model grid, predictor coding, event/continuous
    blocks, design assembly), factored out of the ``__main__`` body so the
    identical coding is unit-testable without ONE. Variable-specific coding lives
    here (project layering: scripts code the predictors): ``side``/``choice`` are
    deviation-coded contra/ipsi relative to ``hemisphere`` and ``contrast`` is
    mean-centered, following ``config.ENCODING_TERMS``.

    Parameters
    ----------
    trials : pd.DataFrame
        Trials table; needs the event-time columns named in ``terms`` plus
        ``stim_side``, ``choice``, ``feedbackType``, ``contrast``,
        ``signed_contrast``, and ``intervals_0``/``intervals_1``.
    target : pd.Series
        Preprocessed photometry signal indexed by time (s); resampled onto the
        model grid as the fit target.
    hemisphere : str
        Recording hemisphere ('l'/'r'); sets the contra/ipsi reference for
        deviation coding.
    expander : Callable[[np.ndarray], np.ndarray]
        Basis expansion applied to each event train (FIR ``lag_expand`` or
        ``raised_cosine_expand`` with bound parameters).
    continuous : dict[str, pd.Series], optional
        Continuous regressors (wheel velocity, pose speed/coordinates) indexed by
        time; each is resampled onto the grid unlagged.
    terms : dict
        Event term spec (default ``config.ENCODING_TERMS``).

    Returns
    -------
    tuple[np.ndarray, dict[str, slice], pd.Series]
        The design matrix, the block-name -> column-span map, and the target
        resampled onto the model grid (indexed by the grid times).
    """
    tvec = make_time_grid(target.index[0], target.index[-1], ENCODING_DT)
    target_grid = pd.Series(interpolate_to_grid(target, tvec), index=tvec)

    coded = task.add_relative_contrast(trials.assign(hemisphere=hemisphere))

    blocks = {}
    for event, spec in terms.items():
        modulators = {name: _code_modulator(name, kind, coded)
                      for name, kind in spec['modulators'].items()}
        split = coded[spec['split_by']] if spec['split_by'] else None
        blocks.update(build_event_blocks(
            coded[event].to_numpy(), tvec, expander,
            modulators=modulators, interactions=spec['interactions'],
            split=split, name=event))

    for name, series in (continuous or {}).items():
        blocks[name] = build_continuous_block(series, tvec)

    design, slices = build_design_matrix(blocks)
    return design, slices, target_grid


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('eid', help='session to fit')
    parser.add_argument('brain_region', help='recording/channel to fit, e.g. SNc-l')
    args = parser.parse_args()

    # --- Select the recording through the group object (single source of truth)
    df = pd.read_parquet(SESSIONS_FPATH)
    one = _get_default_connection()
    group = PhotometrySessionGroup.from_catalog(df, one=one, h5_dir=SESSIONS_H5_DIR)
    group.filter_sessions()
    recording = group.recordings.query(
        'eid == @args.eid and brain_region == @args.brain_region').iloc[0]
    hemisphere = recording['hemisphere']
    ps = group._get_session(recording)

    # --- Load and preprocess this session's data
    ps.load_trials()
    ps.load_photometry()
    ps.preprocess(targets=[args.brain_region])
    target = ps.photometry['GCaMP_preprocessed'][args.brain_region]
    ps.load_wheel()
    ps.load_camera_times()
    ps.load_pose()

    # --- Continuous regressors (variable-specific session wiring): wheel velocity
    #     plus per-keypoint pose speed and raw coordinate traces.
    continuous = {'wheel_velocity': pd.Series(
        ps.wheel['velocity'].to_numpy(), index=ps.wheel['times'].to_numpy())}
    for kp in ENCODING_POSE_KEYPOINTS:
        speed = keypoint_speed(
            ps.pose[f'{kp}_x'].to_numpy(), ps.pose[f'{kp}_y'].to_numpy(),
            ps.pose[f'{kp}_likelihood'].to_numpy())
        continuous[f'{kp}_speed'] = pd.Series(speed, index=ps.pose_times)
        continuous[f'{kp}_x'] = pd.Series(ps.pose[f'{kp}_x'].to_numpy(), index=ps.pose_times)
        continuous[f'{kp}_y'] = pd.Series(ps.pose[f'{kp}_y'].to_numpy(), index=ps.pose_times)

    # --- Build -> fit -> evaluate. FIR basis by default; swap `expander` for
    #     `partial(raised_cosine_expand, tvec=..., ...)` to use the cosine basis.
    #     build_encoding_design codes predictors and assembles the design (steps
    #     4-7); fitting and evaluation stay explicit here.
    lags = make_lags(ENCODING_N_LAGS)
    expander = partial(lag_expand, lags=lags)
    design, slices, target_grid = build_encoding_design(
        ps.trials, target, hemisphere, expander, continuous=continuous)
    fit = fit_encoding_model(
        design, target_grid, slices, ENCODING_ALPHAS, ENCODING_CV,
        label=f'{args.eid} {args.brain_region}')
    deltas = ps.delta_r_squared(fit, cv=ENCODING_CV)

    # --- Inspection plots (one baseline kernel panel per event term)
    fig_dir = PROJECT_ROOT / 'figures/encoding'
    fig_dir.mkdir(parents=True, exist_ok=True)
    tag = f'{args.eid}_{args.brain_region}'

    ax = plot_encoding_prediction(fit)
    ax.figure.savefig(fig_dir / f'{tag}_prediction.svg',
                      dpi=FIGURE_DPI, bbox_inches='tight')

    event_names = [name for name in slices if name.endswith('|baseline')]
    fig = plot_encoding_kernels(fit, event_names, lags)
    fig.savefig(fig_dir / f'{tag}_kernels.svg', dpi=FIGURE_DPI, bbox_inches='tight')

    ax = plot_delta_r_squared(deltas)
    ax.figure.savefig(fig_dir / f'{tag}_delta_r2.svg',
                      dpi=FIGURE_DPI, bbox_inches='tight')
    print(f'Encoding figures saved to {fig_dir}')
