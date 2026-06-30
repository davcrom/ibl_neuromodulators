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
                            for event in ['stimOn_times',
                                          'firstMovement_times',
                                          'feedback_times']:
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


class TestSaveLMMFrames:
    """The pure save step writes exactly the named CSVs, nothing else."""

    def test_writes_named_csvs_only(self, tmp_path):
        from scripts.responses import _save_lmm_frames
        frames = {
            'response_lmm_task_ceiling': pd.DataFrame({
                'target_NM': ['VTA-DA'], 'event': ['stimOn_times'],
                'marginal': [0.1], 'conditional': [0.3]}),
            'response_lmm_task_reliability_cv': pd.DataFrame({
                'target_NM': ['VTA-DA'], 'event': ['stimOn_times'],
                'predictor': ['contrast'], 'fold': ['s0'], 'delta_r2': [0.02]}),
        }
        _save_lmm_frames(frames, tmp_path)

        written = {p.name for p in tmp_path.glob('*.csv')}
        assert written == {'response_lmm_task_ceiling.csv',
                           'response_lmm_task_reliability_cv.csv'}
        ceiling = pd.read_csv(tmp_path / 'response_lmm_task_ceiling.csv')
        assert list(ceiling.columns) == [
            'target_NM', 'event', 'marginal', 'conditional']
        assert ceiling['marginal'].iloc[0] == 0.1


class TestPlotLMMFigures:

    def _run(self, tmp_path):
        from scripts.responses import plot_lmm_figures
        group = _make_movement_group()
        fig_dir = tmp_path / 'lmm'
        fig_dir.mkdir(parents=True, exist_ok=True)
        plot_lmm_figures(group, fig_dir, tmp_path)
        return fig_dir

    def test_writes_suite_csvs_with_consistent_identifiers(self, tmp_path):
        """The orchestration saves the coefficients, ceiling, and both
        reliability frames under the response_lmm naming convention, each
        carrying the (target_NM, event) identifiers and reported quantities."""
        self._run(tmp_path)
        expected = {
            'response_lmm_task_coefficients.csv': {'target_NM', 'event', 'term',
                                                   'Coef.', 'P>|z|'},
            'response_lmm_task_ceiling.csv': {'target_NM', 'event',
                                              'marginal', 'conditional'},
            'response_lmm_task_reliability_cv.csv': {
                'target_NM', 'event', 'predictor', 'fold', 'delta_r2'},
            'response_lmm_task_reliability_jackknife.csv': {
                'target_NM', 'event', 'predictor', 'fold', 'delta_r2'},
        }
        for fname, cols in expected.items():
            df = pd.read_csv(tmp_path / fname)
            assert cols.issubset(df.columns), fname
            assert len(df) > 0, fname

    def test_reliability_predictors_span_main_and_interactions(self, tmp_path):
        """The combined reliability frame carries the drop-one main-effect
        predictors and the omnibus interactions predictor on one axis."""
        self._run(tmp_path)
        df = pd.read_csv(tmp_path / 'response_lmm_task_reliability_cv.csv')
        assert 'interactions' in set(df['predictor'])
        assert {'contrast', 'side', 'reward'} & set(df['predictor'])

    def test_reward_predictor_only_at_feedback(self, tmp_path):
        """Reward is only known at feedback, so the reliability frame carries a
        reward drop-one predictor for ``feedback_times`` alone; stimOn and
        firstMovement carry only contrast, side, and interactions."""
        self._run(tmp_path)
        df = pd.read_csv(tmp_path / 'response_lmm_task_reliability_cv.csv')
        reward_events = set(df[df['predictor'] == 'reward']['event'])
        assert reward_events == {'feedback_times'}
        for event in ['stimOn_times', 'firstMovement_times']:
            preds = set(df[df['event'] == event]['predictor'])
            assert 'reward' not in preds
            assert {'contrast', 'side', 'interactions'} <= preds

    def test_renders_labelled_summary_figures(self, tmp_path):
        fig_dir = self._run(tmp_path)
        assert any(fig_dir.glob('response_lmm_task_summary_*.svg'))


class TestPlotMovementFigures:

    def _run(self, tmp_path):
        from scripts.responses import plot_movement_figures
        group = _make_movement_group()
        fig_dirs = {'movement_model_comparison': tmp_path / 'model_comparison'}
        for d in fig_dirs.values():
            d.mkdir(parents=True, exist_ok=True)
        plot_movement_figures(group, fig_dirs, tmp_path)
        return fig_dirs

    def test_writes_reliability_and_r2_csvs(self, tmp_path):
        self._run(tmp_path)
        for fname, cols in [
            ('response_lmm_movement_reliability_cv.csv',
             {'target_NM', 'event', 'predictor', 'fold', 'delta_r2',
              'movement_var'}),
            ('response_lmm_movement_reliability_jackknife.csv',
             {'target_NM', 'event', 'predictor', 'fold', 'delta_r2',
              'movement_var'}),
            ('response_lmm_movement_r2.csv',
             {'target_NM', 'event', 'name', 'marginal_r2', 'movement_var'}),
            ('response_lmm_movement_ceiling.csv',
             {'target_NM', 'event', 'marginal', 'conditional'}),
        ]:
            df = pd.read_csv(tmp_path / fname)
            assert cols.issubset(df.columns), fname

    def test_reliability_predictors_extend_task_set(self, tmp_path):
        """The movement reliability axis carries the task drop-one predictors
        plus the movement predictor on one axis."""
        self._run(tmp_path)
        df = pd.read_csv(tmp_path / 'response_lmm_movement_reliability_cv.csv')
        preds = set(df['predictor'])
        assert 'movement' in preds
        assert 'interactions' in preds
        assert {'contrast', 'side', 'reward'} & preds

    def test_renders_movement_figures(self, tmp_path):
        fig_dirs = self._run(tmp_path)
        assert any(fig_dirs['movement_model_comparison']
                   .glob('response_lmm_movement_reliability_*.svg'))
        assert any(fig_dirs['movement_model_comparison']
                   .glob('response_lmm_movement_r2_*.svg'))
        assert (fig_dirs['movement_model_comparison']
                / 'response_lmm_movement_ceiling.svg').exists()


class TestPlotPersessionFigures:
    """The persession figure step plots from the in-scope drop-one frame
    (``response_ols_dropone_results``) without recomputing or writing data."""

    def _stub_frame(self):
        predictors = ['contrast', 'side', 'reward', 'choice_side',
                      'log_reaction_time', 'peak_velocity']
        rows = []
        for s in range(3):
            for p in predictors:
                rows.append({
                    'eid': f'e{s}', 'subject': f's{s}', 'target_NM': 'VTA-DA',
                    'brain_region': 'VTA', 'event': 'stimOn_times',
                    'predictor': p, 'r2': 0.5, 'delta_r2': 0.05,
                    'n_trials': 100})
        return pd.DataFrame(rows)

    def test_plots_from_loaded_frame_without_recompute(self, tmp_path):
        from scripts import responses
        group = MagicMock()
        group.response_ols_dropone_results = self._stub_frame()

        fig_dir = tmp_path / 'persession'
        fig_dir.mkdir()
        responses.plot_persession_figures(group, fig_dir)

        group.response_ols_dropone.assert_not_called()
        svg = fig_dir / 'response_ols_persession_dropone.svg'
        assert svg.exists() and svg.stat().st_size > 0


def _responses_source():
    """Return (full source, __main__ block) of scripts/responses.py."""
    from pathlib import Path
    import scripts.responses
    src = Path(scripts.responses.__file__).read_text()
    main_block = src.split("if __name__ == '__main__':", 1)[1]
    return src, main_block


class TestReprocessWiring:
    """Source-level wiring: cache-load is default, --reprocess opts into refit,
    and the trial-level LMM/movement stages are no longer called from __main__
    while their definitions remain."""

    def test_reprocess_flag_replaces_plot(self):
        src, main_block = _responses_source()
        assert "'--reprocess'" in src
        assert "'--plot'" not in src
        assert 'args.reprocess' in main_block

    def test_main_block_drops_lmm_and_movement_calls(self):
        _, main_block = _responses_source()
        assert 'plot_lmm_figures(' not in main_block
        assert 'plot_movement_figures(' not in main_block

    def test_lmm_and_movement_definitions_remain(self):
        src, _ = _responses_source()
        assert 'def plot_lmm_figures(' in src
        assert 'def plot_movement_figures(' in src


def _reprocess_and_default_branches():
    """Split __main__ into the (reprocess body, default-onward) text."""
    _, main_block = _responses_source()
    reprocess, default = main_block.split('\n    else:', 1)
    return reprocess, default


class TestVarcompWiring:
    """Source-level wiring: --reprocess fits and caches the variance-components
    tables, default mode loads them, and the violin figure is plotted in both."""

    def test_reprocess_fits_and_caches_varcomp(self):
        reprocess, _ = _reprocess_and_default_branches()
        assert 'response_varcomp(' in reprocess
        assert 'RESPONSE_VARCOMP_SUMMARY_FPATH' in reprocess
        assert 'RESPONSE_VARCOMP_VIOLIN_FPATH' in reprocess
        assert reprocess.count('.to_parquet(') >= 2

    def test_default_loads_cached_varcomp(self):
        _, default = _reprocess_and_default_branches()
        assert 'load_response_varcomp_summary(' in default
        assert 'load_response_varcomp_violin(' in default

    def test_violin_figure_plotted(self):
        _, main_block = _responses_source()
        assert 'plot_varcomp_violins(' in main_block
        assert 'response_varcomp_violins.svg' in main_block
