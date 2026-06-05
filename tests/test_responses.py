"""Tests for scripts/responses.py movement-encoding wiring."""
from unittest.mock import MagicMock

import numpy as np
import pandas as pd


def _make_group(response_magnitudes, trial_regressors):
    """Build a PhotometrySessionGroup with injected modeling frames."""
    from iblnm.data import PhotometrySessionGroup
    recs = pd.DataFrame([{
        'eid': eid, 'subject': subj,
        'brain_region': tnm.split('-')[0], 'hemisphere': 'r',
        'target_NM': tnm, 'NM': tnm.split('-')[1],
        'session_type': 'biased', 'start_time': '2024-01-01T10:00:00',
        'number': 1, 'task_protocol': 'biased_protocol',
    } for eid, subj, tnm in
        response_magnitudes[['eid', 'subject', 'target_NM']]
        .drop_duplicates().itertuples(index=False)])
    group = PhotometrySessionGroup(recs, one=MagicMock())
    group.response_magnitudes = response_magnitudes
    group.trial_regressors = trial_regressors
    return group


def _make_movement_group(n_per_cell=50, seed=0):
    """Synthetic group: 2 target_NMs, 3 subjects, 3 contrasts, both events."""
    rng = np.random.default_rng(seed)
    resp_rows, reg_rows = [], []
    trial = 0
    for tnm in ['VTA-DA', 'DR-5HT']:
        for s in range(3):
            eid = f'eid-{tnm}-{s}'
            subj_slope = 0.4 + rng.normal(0, 0.1)
            for contrast in [0.0, 25.0, 100.0]:
                for stim_side in ['left', 'right']:
                    for fb in [1, -1]:
                        for _ in range(n_per_cell):
                            log_rt = rng.normal(-0.7, 0.3)
                            response = (0.3 * (contrast / 100)
                                        + subj_slope * log_rt
                                        + rng.normal(0, 0.5))
                            for event in ['stimOn_times', 'feedback_times']:
                                resp_rows.append({
                                    'eid': eid, 'subject': f'subj-{s}',
                                    'target_NM': tnm, 'NM': tnm.split('-')[1],
                                    'brain_region': tnm.split('-')[0],
                                    'hemisphere': 'r', 'event': event,
                                    'trial': trial, 'response': response,
                                })
                            reg_rows.append({
                                'eid': eid, 'trial': trial,
                                'signed_contrast': (contrast if stim_side == 'right'
                                                    else -contrast),
                                'contrast': contrast, 'stim_side': stim_side,
                                'choice': rng.choice([-1, 1]), 'feedbackType': fb,
                                'probabilityLeft': 0.5,
                                'reaction_time': 10 ** log_rt,
                                'movement_time': abs(rng.normal(0.3, 0.1)),
                                'response_time': abs(rng.normal(0.8, 0.2)) + 0.1,
                                'peak_velocity': abs(rng.normal(5.0, 2.0)),
                            })
                            trial += 1
    return _make_group(pd.DataFrame(resp_rows), pd.DataFrame(reg_rows))


class TestBuildMovementDF:

    def _base_frames(self):
        resp = pd.DataFrame([
            {'eid': 'e0', 'subject': 's0', 'target_NM': 'VTA-DA', 'NM': 'DA',
             'brain_region': 'VTA', 'hemisphere': 'r', 'event': 'stimOn_times',
             'trial': t, 'response': 1.0}
            for t in range(4)
        ])
        reg = pd.DataFrame([
            # valid
            {'eid': 'e0', 'trial': 0, 'signed_contrast': 0.25, 'contrast': 25.0,
             'stim_side': 'right', 'choice': 1, 'feedbackType': 1,
             'probabilityLeft': 0.5, 'reaction_time': 0.2, 'movement_time': 0.3,
             'response_time': 0.8, 'peak_velocity': 5.0},
            # biased block -> dropped
            {'eid': 'e0', 'trial': 1, 'signed_contrast': 0.25, 'contrast': 25.0,
             'stim_side': 'right', 'choice': 1, 'feedbackType': 1,
             'probabilityLeft': 0.8, 'reaction_time': 0.2, 'movement_time': 0.3,
             'response_time': 0.8, 'peak_velocity': 5.0},
            # nogo -> dropped
            {'eid': 'e0', 'trial': 2, 'signed_contrast': 0.25, 'contrast': 25.0,
             'stim_side': 'right', 'choice': 0, 'feedbackType': 1,
             'probabilityLeft': 0.5, 'reaction_time': 0.2, 'movement_time': 0.3,
             'response_time': 0.8, 'peak_velocity': 5.0},
            # fast response -> dropped
            {'eid': 'e0', 'trial': 3, 'signed_contrast': 0.25, 'contrast': 25.0,
             'stim_side': 'right', 'choice': 1, 'feedbackType': 1,
             'probabilityLeft': 0.5, 'reaction_time': 0.2, 'movement_time': 0.3,
             'response_time': 0.01, 'peak_velocity': 5.0},
        ])
        return resp, reg

    def test_filters_to_valid_trials(self):
        from scripts.responses import build_movement_df
        resp, reg = self._base_frames()
        df = build_movement_df(_make_group(resp, reg))
        assert list(df['trial']) == [0]

    def test_adds_log_columns(self):
        from scripts.responses import build_movement_df
        resp, reg = self._base_frames()
        df = build_movement_df(_make_group(resp, reg))
        for var in ['reaction_time', 'movement_time', 'peak_velocity']:
            assert f'log_{var}' in df.columns
        np.testing.assert_allclose(
            df['log_reaction_time'].iloc[0], np.log10(0.2))

    def test_nonpositive_timing_gives_nan_log(self):
        from scripts.responses import build_movement_df
        resp, reg = self._base_frames()
        reg.loc[reg['trial'] == 0, 'peak_velocity'] = 0.0
        df = build_movement_df(_make_group(resp, reg))
        assert df['log_peak_velocity'].isna().all()

    def test_retains_baseline_stimon_firstmovement_events(self):
        """The movement DV set is baseline, stimOn, firstMovement; feedback is
        excluded (no longer filtered to stimOn only)."""
        from scripts.responses import build_movement_df
        events = ['baseline', 'stimOn_times', 'firstMovement_times',
                  'feedback_times']
        resp = pd.DataFrame([
            {'eid': 'e0', 'subject': 's0', 'target_NM': 'VTA-DA', 'NM': 'DA',
             'brain_region': 'VTA', 'hemisphere': 'r', 'event': event,
             'trial': 0, 'response': 1.0}
            for event in events
        ])
        reg = pd.DataFrame([
            {'eid': 'e0', 'trial': 0, 'signed_contrast': 0.25, 'contrast': 25.0,
             'stim_side': 'right', 'choice': 1, 'feedbackType': 1,
             'probabilityLeft': 0.5, 'reaction_time': 0.2, 'movement_time': 0.3,
             'response_time': 0.8, 'peak_velocity': 5.0},
        ])
        df = build_movement_df(_make_group(resp, reg))
        assert set(df['event']) == {
            'baseline', 'stimOn_times', 'firstMovement_times'}


class TestPlotMovementFigures:

    def test_writes_csvs_and_figures(self, tmp_path):
        from scripts.responses import plot_movement_figures
        group = _make_movement_group()
        fig_dirs = {
            'movement_descriptive': tmp_path / 'descriptive',
            'movement_model_comparison': tmp_path / 'model_comparison',
            'movement_slopes': tmp_path / 'slopes',
        }
        for d in fig_dirs.values():
            d.mkdir(parents=True, exist_ok=True)

        plot_movement_figures(group, fig_dirs, tmp_path)

        assert (tmp_path / 'jackknife_model_comparison.csv').exists()
        assert (tmp_path / 'movement_marginal_r2.csv').exists()
        assert any(fig_dirs['movement_descriptive'].glob('*.svg'))
