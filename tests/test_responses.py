"""Tests for scripts/responses.py movement-encoding wiring."""
from contextlib import ExitStack
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from iblnm.analysis import select_modeling_trials
from iblnm.config import STIM_ONSET_EVENT
from iblnm.data import PhotometrySession


def _make_group(magnitudes, regressors):
    """Build a group over ``magnitudes``, beside the merged frame it covers.

    The two frames are given as the store produces them — magnitudes per
    recording x event x trial, regressors per session x trial — and merged
    here, because the pipeline's own frame is merged and the hemisphere-relative
    ``side`` / ``choice_side`` are derived on it.
    """
    from iblnm.data import PhotometrySessionGroup
    from iblnm.task import add_relative_contrast
    recs = pd.DataFrame([{
        'eid': eid, 'subject': subj,
        'brain_region': tnm.split('-')[0], 'hemisphere': 'r',
        'target_NM': tnm, 'NM': tnm.split('-')[1],
        'session_type': 'biased', 'start_time': '2024-01-01T10:00:00',
        'number': 1, 'task_protocol': 'biased_protocol',
    } for eid, subj, tnm in
        magnitudes[['eid', 'subject', 'target_NM']]
        .drop_duplicates().itertuples(index=False)])
    group = PhotometrySessionGroup(recs, one=MagicMock())
    merged = add_relative_contrast(
        magnitudes.merge(regressors, on=['eid', 'trial'], how='left'))
    return group, merged


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
                            for event in ['stimOnTrigger_times',
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


class _TraceSession:
    """Session stub carrying the real correction methods over injected data.

    ``mask_subsequent_events`` reads ``trials`` and ``subtract_baseline`` reads
    nothing but its argument, so the correction under test is the pipeline's
    own, not a re-implementation.
    """

    mask_subsequent_events = PhotometrySession.mask_subsequent_events
    subtract_baseline = PhotometrySession.subtract_baseline

    def __init__(self, trials, photometry_responses):
        self.trials = trials
        self.photometry_responses = photometry_responses


class TestConditionTraces:
    """The plotting pass's correction-and-aggregation step: per-trial traces
    in, per-condition means and SEMs out."""

    # Five samples spanning the (-0.1, 0) baseline window, which covers the
    # -0.1 sample alone, so a hand-computed baseline is one number per trial.
    _TIMES = np.array([-0.2, -0.1, 0.0, 0.1, 0.2])

    def _recording(self, traces, feedback_lags=None, eid='e1', subject='m1'):
        """One ``(recording row, session)`` pair holding the given traces.

        ``traces`` is one row per trial, one column per entry of ``_TIMES``.
        ``feedback_lags`` places each trial's feedback relative to its stimulus
        onset; the default puts it past the last sample, so nothing is masked.
        """
        n = len(traces)
        if feedback_lags is None:
            feedback_lags = [10.0] * n
        onsets = np.arange(n, dtype=float) * 100
        trials = pd.DataFrame({
            'trial': np.arange(n),
            'stimOnTrigger_times': onsets,
            'feedback_times': onsets + np.asarray(feedback_lags, dtype=float),
        })
        responses = xr.DataArray(
            np.asarray(traces, dtype=float)[None, :, :],
            dims=['event', 'trial', 'time'],
            coords={'event': [STIM_ONSET_EVENT], 'trial': np.arange(n),
                    'time': self._TIMES})
        rec = pd.Series({'eid': eid, 'subject': subject, 'brain_region': 'VTA',
                         'hemisphere': 'r', 'target_NM': 'VTA-DA'})
        return rec, _TraceSession(trials, {'VTA': responses})

    def _trials(self, n, eid='e1', subject='m1', contrast=100.0,
                feedback_type=1, probability_left=0.5):
        """The merged trial frame the selection runs on, one row per trial."""
        return pd.DataFrame({
            'eid': eid, 'subject': subject, 'target_NM': 'VTA-DA',
            'brain_region': 'VTA', 'event': STIM_ONSET_EVENT,
            'trial': np.arange(n), 'response': 1.0,
            'contrast': contrast, 'feedbackType': feedback_type,
            'choice': 1, 'response_time': 1.0, 'reaction_time': 0.2,
            'probabilityLeft': probability_left,
        })

    def test_uncorrected_means_are_the_raw_trace_means(self):
        """With the correction off, each condition's mean at each time point is
        the plain trial mean of the traces as stored."""
        from scripts.responses import condition_traces
        rec, ps = self._recording([[0., 1., 2., 3., 4.],
                                   [2., 3., 4., 5., 6.]])
        agg = condition_traces([(rec, ps)], self._trials(2), mode='pool',
                               correct=False)

        assert agg['mean'].tolist() == [1., 2., 3., 4., 5.]
        assert agg['time'].tolist() == self._TIMES.tolist()
        assert (agg['n'] == 2).all()

    def test_correction_masks_the_next_event_and_subtracts_the_baseline(self):
        """With the correction on, each trial loses the samples past its
        feedback and is shifted by its own ``BASELINE_WINDOW`` mean — here the
        single sample at -0.1 s. The first trial's feedback lands at 0.15 s, so
        its 0.2 s sample is masked and only the second trial is averaged there.
        """
        from scripts.responses import condition_traces
        rec, ps = self._recording([[0., 1., 2., 3., 4.],
                                   [2., 3., 4., 5., 6.]],
                                  feedback_lags=[0.15, 10.0])
        agg = condition_traces([(rec, ps)], self._trials(2), mode='pool',
                               correct=True)

        assert agg['mean'].tolist() == [-1., 0., 1., 2., 3.]
        assert agg['n'].tolist() == [2, 2, 2, 2, 1]

    def _uneven_cohort(self):
        """Three recordings, flat traces, deliberately unbalanced.

        Subject m1 records twice — 4 trials at 0 and 2 trials at 2 — and
        subject m2 once, 1 trial at 10. Each mode therefore lands on its own
        number: trials weight m1's 6 trials, subjects weight the two mice
        equally, and recordings weight the three fibers equally.
        """
        recordings = [self._recording([[0.] * 5] * 4, eid='e1', subject='m1'),
                      self._recording([[2.] * 5] * 2, eid='e2', subject='m1'),
                      self._recording([[10.] * 5], eid='e3', subject='m2')]
        trials = pd.concat([self._trials(4, eid='e1', subject='m1'),
                            self._trials(2, eid='e2', subject='m1'),
                            self._trials(1, eid='e3', subject='m2')],
                           ignore_index=True)
        return recordings, trials

    @pytest.mark.parametrize('mode, expected, n', [
        ('pool', 14 / 7, 7),               # every trial a unit
        ('subject', (4 / 6 + 10) / 2, 2),  # mean of the two subject means
        ('subject_centered', 4.0, 3),      # mean of the three recording means
    ])
    def test_each_mode_weights_its_own_unit(self, mode, expected, n):
        """The same traces reduce to a different mean under each mode, and ``n``
        reports the units the SEM was taken over."""
        from scripts.responses import condition_traces
        recordings, trials = self._uneven_cohort()

        agg = condition_traces(recordings, trials, mode=mode, correct=False)

        assert agg['mean'].tolist() == pytest.approx([expected] * 5)
        assert agg['n'].tolist() == [n] * 5

    def test_averages_the_trials_the_models_are_fitted_on(self):
        """The trial set is ``select_modeling_trials``', not the unbiased block:
        a ``probabilityLeft`` 0.8 trial is averaged and a no-go trial is not."""
        from scripts.responses import condition_traces
        rec, ps = self._recording([[0.] * 5, [10.] * 5, [100.] * 5])
        trials = self._trials(3)
        trials['probabilityLeft'] = [0.5, 0.8, 0.5]
        trials['choice'] = [1, 1, 0]

        agg = condition_traces([(rec, ps)], trials, mode='pool', correct=False)

        assert agg['mean'].tolist() == [5.] * 5
        assert agg['n'].tolist() == [2] * 5


class TestSaveLMMFrames:
    """The pure save step writes exactly the named CSVs, nothing else."""

    def test_writes_named_csvs_only(self, tmp_path):
        from scripts.responses import _save_lmm_frames
        frames = {
            'response_lmm_task_ceiling': pd.DataFrame({
                'target_NM': ['VTA-DA'], 'event': ['stimOnTrigger_times'],
                'marginal': [0.1], 'conditional': [0.3]}),
            'response_lmm_task_reliability_cv': pd.DataFrame({
                'target_NM': ['VTA-DA'], 'event': ['stimOnTrigger_times'],
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
        group, magnitudes = _make_movement_group()
        fig_dir = tmp_path / 'lmm'
        fig_dir.mkdir(parents=True, exist_ok=True)
        plot_lmm_figures(group, magnitudes, fig_dir, tmp_path)
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
        reward drop-one predictor for ``feedback_times`` alone; stimOn carries
        only contrast, side, and interactions."""
        self._run(tmp_path)
        df = pd.read_csv(tmp_path / 'response_lmm_task_reliability_cv.csv')
        reward_events = set(df[df['predictor'] == 'reward']['event'])
        assert reward_events == {'feedback_times'}
        preds = set(df[df['event'] == 'stimOnTrigger_times']['predictor'])
        assert 'reward' not in preds
        assert {'contrast', 'side', 'interactions'} <= preds

    def test_renders_labelled_summary_figures(self, tmp_path):
        fig_dir = self._run(tmp_path)
        assert any(fig_dir.glob('response_lmm_task_summary_*.svg'))


class TestPlotMovementFigures:

    def _run(self, tmp_path):
        from scripts.responses import plot_movement_figures
        group, magnitudes = _make_movement_group()
        fig_dirs = {'movement_model_comparison': tmp_path / 'model_comparison'}
        for d in fig_dirs.values():
            d.mkdir(parents=True, exist_ok=True)
        plot_movement_figures(group, magnitudes, fig_dirs, tmp_path)
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


class _LinkSession:
    """Session stub exposing what the two link functions call and read.

    The link functions are the only place a session's state becomes group
    data, so what they are responsible for is calling the right method and
    handing back the right attributes — not the modelling itself, which is
    covered against real sessions in ``tests/test_data.py``. This stub records
    the arguments it was passed and plants the attributes the real methods
    assign.
    """

    def __init__(self):
        self.donor_frame = pd.DataFrame({'trial': [0, 1], 'contrast': [1.0, 2.0]})
        self.response_magnitudes = pd.DataFrame({'trial': [0, 1],
                                                 'response': [0.5, 0.7]})
        self.response_ols = pd.DataFrame({'predictor': ['contrast'],
                                          'delta_r2': [0.04]})
        self.prepared = None
        self.fitted = None

    def prepare_donor_frame(self):
        self.prepared = True
        return self.donor_frame

    def fit_responses(self, formulas, donors):
        self.fitted = (formulas, donors)
        return self.response_ols


class TestLinkFunctions:
    """The two module-level functions ``group.process`` maps over the sessions:
    each calls one session method and returns plain data, holding no reference
    to a group."""

    def test_build_donor_frame_returns_that_sessions_donor_frame(self):
        from scripts.responses import build_donor_frame
        ps = _LinkSession()

        returned = build_donor_frame(ps)

        assert ps.prepared is True
        assert returned is ps.donor_frame

    def test_fit_session_forwards_its_arguments_and_returns_both_frames(self):
        from scripts.responses import fit_session
        ps = _LinkSession()
        formulas = {'full': '{response} ~ contrast'}
        donors = {'eid-1': 'donor'}

        magnitudes, fits = fit_session(ps, formulas, donors)

        assert ps.fitted == (formulas, donors)
        assert magnitudes is ps.response_magnitudes
        assert fits is ps.response_ols


class TestVarcompCoefficients:
    """The variance-components stage takes its per-session coefficients out of
    the merged per-recording OLS frame, rather than a coefficients frame of its
    own."""

    _REGRESSORS = ['contrast', 'side']

    def _coefficients(self):
        """The coefficients frame the varcomp stage used to be fed directly."""
        return pd.DataFrame([
            {'eid': eid, 'subject': 'm1', 'target_NM': 'VTA-DA',
             'brain_region': region, 'event': 'feedback_times',
             'regressor': regressor, 'coef': coef, 'coef_se': 0.1,
             'n_trials': 200}
            for eid, region in [('e1', 'VTA'), ('e2', 'SNc')]
            for coef, regressor in zip([0.3, -0.4], self._REGRESSORS)
        ])

    def _ols_persession(self):
        """The same weights as ticket-12's merged frame carries them: keyed by
        ``predictor``, with the drop-one and significance columns alongside."""
        return pd.DataFrame([
            {'eid': eid, 'subject': 'm1', 'target_NM': 'VTA-DA',
             'brain_region': region, 'event': 'feedback_times',
             'predictor': regressor, 'n_trials': 200, 'r2_full': 0.4,
             'r2_full_adj': 0.35, 'delta_r2': 0.05, 'delta_r2_adj': 0.03,
             'delta_r2_null_median': 0.01, 'coef': coef, 'coef_se': 0.1,
             'p_value': 0.01, 'q_value': 0.02, 'n_donors': 700}
            for eid, region in [('e1', 'VTA'), ('e2', 'SNc')]
            for coef, regressor in zip([0.3, -0.4], self._REGRESSORS)
        ])

    def test_view_matches_the_standalone_coefficients_frame(self):
        """The view is the coefficients frame the stage used to read: same
        columns, same grain, same weights — so the posteriors it feeds are
        unchanged."""
        from scripts.responses import varcomp_coefficients
        pd.testing.assert_frame_equal(
            varcomp_coefficients(self._ols_persession()),
            self._coefficients())


class TestComputeMaskingDiagnostics:
    """How much of the response window the masking removed, per trial type and
    cohort, reported alongside every contrast-dependent result."""

    _WINDOW = (0.1, 0.35)

    def _magnitudes(self, masked_fractions, reaction_times=None):
        """One recording-event cell, one trial per entry of the given lists.

        Every trial is a go trial with a real response time and a
        non-negative reaction time, so nothing is dropped by the modeling
        selection and the trial count equals the list length.
        """
        n = len(masked_fractions)
        if reaction_times is None:
            reaction_times = [0.2] * n
        magnitudes = pd.DataFrame({
            'eid': 'e1', 'subject': 'm1', 'session_type': 'biased',
            'NM': 'DA', 'target_NM': 'VTA-DA', 'brain_region': 'VTA',
            'hemisphere': 'r', 'event': 'stimOnTrigger_times',
            'trial': np.arange(n), 'response': 1.0,
            'masked_fraction': masked_fractions,
        })
        regressors = pd.DataFrame({
            'eid': 'e1', 'trial': np.arange(n), 'contrast': 100.0,
            'stim_side': 'right', 'signed_contrast': 100.0,
            'feedbackType': 1, 'choice': 1, 'response_time': 1.0,
            'reaction_time': reaction_times, 'probabilityLeft': 0.5,
        })
        return _make_group(magnitudes, regressors)[1]

    def _cell(self, masked_fractions, reaction_times=None):
        from scripts.responses import compute_masking_diagnostics
        frame = compute_masking_diagnostics(
            self._magnitudes(masked_fractions, reaction_times),
            window=self._WINDOW)
        assert len(frame) == 1
        return frame.iloc[0]

    def test_partially_masked_cell(self):
        """Half the trials lose half their window: a quarter of the window is
        masked on average, over half the trials, none of them end to end."""
        cell = self._cell([0.5] * 5 + [0.0] * 5)
        assert cell['n_trials'] == 10
        assert cell['masked_fraction_mean'] == pytest.approx(0.25)
        assert cell['pct_any_masked'] == pytest.approx(50.0)
        assert cell['pct_fully_masked'] == pytest.approx(0.0)

    def test_fully_masked_trial_counts_as_masked_too(self):
        """A window masked end to end is one of the trials with any masking."""
        cell = self._cell([1.0, 0.0, 0.0, 0.0])
        assert cell['pct_any_masked'] == pytest.approx(25.0)
        assert cell['pct_fully_masked'] == pytest.approx(25.0)

    def test_movement_inside_the_window_is_counted_by_reaction_time(self):
        """First movement at 0.2 s falls inside a (0.1, 0.35) window; one at
        0.5 s does not."""
        cell = self._cell([0.0] * 4, reaction_times=[0.2, 0.2, 0.5, 0.5])
        assert cell['pct_move_in_window'] == pytest.approx(50.0)

    def test_cells_split_by_target_event_contrast_and_feedback(self):
        """The frame's grain: one row per (target_NM, event, contrast,
        feedbackType), carrying the schema's columns."""
        from scripts.responses import compute_masking_diagnostics
        from iblnm.config import MASKING_DIAGNOSTIC_COLUMNS
        magnitudes = self._magnitudes([0.0] * 4)
        magnitudes['contrast'] = [100.0, 100.0, 0.0, 0.0]
        magnitudes['feedbackType'] = [1, -1, 1, -1]
        frame = compute_masking_diagnostics(magnitudes, window=self._WINDOW)
        assert list(frame.columns) == MASKING_DIAGNOSTIC_COLUMNS
        assert len(frame) == 4
        assert (frame['n_trials'] == 1).all()


class TestPlotTraceFigures:
    """The event-triggered averages: one figure per cohort, from the store."""

    def _group(self):
        """Group stub whose two recordings are one cohort each."""
        group = MagicMock()
        group.recordings = pd.DataFrame({
            'eid': ['e1', 'e2'], 'subject': ['m1', 'm2'],
            'brain_region': ['VTA', 'DR'], 'hemisphere': ['r', 'r'],
            'target_NM': ['VTA-DA', 'DR-5HT']})
        return group

    def _aggregate(self):
        return pd.DataFrame([
            {'target_NM': 'VTA-DA', 'event': STIM_ONSET_EVENT,
             'contrast': 100.0, 'feedbackType': 1, 'time': t, 'mean': 0.5,
             'sem': 0.1, 'n': 4} for t in (-0.1, 0.0, 0.1)])

    def test_reads_each_file_once_and_skips_regions_without_a_cut(self,
                                                                  tmp_path):
        """A two-region session is read once and yields both its recordings;
        a region carrying no stored cut yields nothing, as in the modelling
        pass, rather than aborting the cohort."""
        from scripts.responses import _cohort_recordings
        (tmp_path / 'e1.h5').touch()
        session = MagicMock(filepath=tmp_path / 'e1.h5')
        session.photometry_responses = {'VTA': 'cut'}
        group = MagicMock(_sessions={'e1': session})
        group._get_session.return_value = session
        cohort = pd.DataFrame({'eid': ['e1', 'e1'],
                               'brain_region': ['VTA', 'SNc'],
                               'target_NM': ['VTA-DA', 'SNc-DA']})

        yielded = list(_cohort_recordings(group, cohort))

        assert [rec['brain_region'] for rec, _ in yielded] == ['VTA']
        assert session.load_h5.call_count == 1

    def test_one_figure_per_cohort(self, tmp_path):
        from scripts import responses
        with patch.object(responses, 'condition_traces',
                          return_value=self._aggregate()):
            responses.plot_trace_figures(
                self._group(), pd.DataFrame(), tmp_path)

        assert {p.name for p in tmp_path.glob('*.svg')} == {
            'VTA-DA_traces.svg', 'DR-5HT_traces.svg'}

    def test_insets_only_where_configured(self, tmp_path):
        """The small-response cohorts named in ``TRACE_INSET_TARGETNMS`` get
        their panels redrawn on their own scale; the others do not."""
        from scripts import responses
        from iblnm.config import TRACE_INSET_TARGETNMS
        with ExitStack() as stack:
            stack.enter_context(patch.object(responses, 'condition_traces',
                                             return_value=self._aggregate()))
            drawer = stack.enter_context(
                patch.object(responses, 'plot_mean_response_traces'))
            responses.plot_trace_figures(
                self._group(), pd.DataFrame(), tmp_path)

        insets = {call.args[1]: call.kwargs['inset']
                  for call in drawer.call_args_list}
        assert insets == {'VTA-DA': False,
                          'DR-5HT': 'DR-5HT' in TRACE_INSET_TARGETNMS}


class TestPlotResponseFigures:
    """The contrast curves: aggregated in the script, drawn by ``vis``."""

    def test_one_file_per_target_event_and_mode(self, tmp_path):
        from scripts.responses import plot_response_figures
        plot_response_figures(_make_movement_group(n_per_cell=5)[1], tmp_path)

        names = {p.name for p in tmp_path.glob('*.svg')}
        assert names == {
            f'{target_nm}_{event}_response{suffix}.svg'
            for target_nm in ('VTA-DA', 'DR-5HT')
            for event in ('stimOnTrigger', 'firstMovement', 'feedback')
            for suffix in ('_pool', '_subject')}

    def test_hands_the_drawer_pooled_condition_means(self, tmp_path):
        """The frame ``plot_relative_contrast`` receives is the pooled mean of
        that cohort-event's trials, one row per (side, contrast, outcome)."""
        from scripts import responses
        _, magnitudes = _make_movement_group(n_per_cell=5)
        trials = select_modeling_trials(magnitudes)

        with patch.object(responses, 'plot_relative_contrast') as drawer:
            responses.plot_response_figures(magnitudes, tmp_path,
                                            modes=('pool',))

        agg_df, target_nm, event = drawer.call_args_list[0].args
        cell = trials[(trials['target_NM'] == target_nm)
                      & (trials['event'] == event)]
        expected = (cell.groupby(['side', 'contrast', 'feedbackType'])
                    ['response'].mean().sort_index())
        drawn = (agg_df.set_index(['side', 'contrast', 'feedbackType'])
                 ['mean'].sort_index())
        pd.testing.assert_series_equal(drawn, expected, check_names=False)


class TestPlotMaskingFigures:
    """The masking diagnostics save loop: one figure per cohort-event."""

    def test_one_file_per_target_and_event(self, tmp_path):
        from scripts.responses import plot_masking_figures
        diagnostics = pd.DataFrame([
            {'target_NM': target_nm, 'event': event, 'contrast': 100.0,
             'feedbackType': 1, 'n_trials': 50, 'masked_fraction_mean': 0.2,
             'pct_any_masked': 20.0, 'pct_fully_masked': 5.0,
             'pct_move_in_window': 60.0}
            for target_nm in ['VTA-DA', 'DR-5HT']
            for event in ['stimOnTrigger_times', 'feedback_times']
        ])
        plot_masking_figures(diagnostics, tmp_path)
        assert {p.name for p in tmp_path.glob('*.svg')} == {
            'VTA-DA_stimOnTrigger_masking.svg', 'VTA-DA_feedback_masking.svg',
            'DR-5HT_stimOnTrigger_masking.svg', 'DR-5HT_feedback_masking.svg'}


class TestPlotPersessionFigures:
    """The persession figure step plots from the merged OLS frame it is
    handed, at the grain the caller read it in."""

    def _stub_frame(self):
        predictors = ['contrast', 'side', 'reward', 'choice_side',
                      'log_reaction_time', 'peak_velocity']
        rows = []
        for s in range(3):
            for p in predictors:
                rows.append({
                    'eid': f'e{s}', 'subject': f's{s}', 'target_NM': 'VTA-DA',
                    'brain_region': 'VTA', 'event': 'stimOnTrigger_times',
                    'predictor': p, 'n_trials': 100, 'r2_full': 0.5,
                    'r2_full_adj': 0.45, 'delta_r2': 0.05,
                    'delta_r2_adj': 0.03, 'delta_r2_null_median': 0.01,
                    'coef': 0.2, 'coef_se': 0.05, 'p_value': 0.01,
                    'q_value': 0.02, 'n_donors': 700})
        return pd.DataFrame(rows)

    def test_plots_from_a_bare_frame(self, tmp_path):
        """The frame and the output directory are the whole input — no group.
        The per-mouse table is absent here, so this exercises the real drop-one
        figure reading a frame rather than a mock."""
        from scripts import responses

        fig_dir = tmp_path / 'persession'
        fig_dir.mkdir()
        responses.plot_persession_figures(self._stub_frame(), None, fig_dir)

        svg = fig_dir / 'response_ols_persession_dropone.svg'
        assert svg.exists() and svg.stat().st_size > 0

    @pytest.mark.parametrize('display, dropone_name, total_r2_name', [
        ('session', 'plot_ols_dropone', 'plot_ols_total_r2'),
        ('subject', 'plot_ols_dropone_subject', 'plot_ols_total_r2_subject'),
        ('target', 'plot_ols_dropone_violin', 'plot_ols_total_r2_violin'),
    ])
    def test_display_maps_to_function_pair(self, display, dropone_name,
                                           total_r2_name):
        """The dispatch table pairs each display mode with its matching
        (drop-one, full-model R²) vis functions."""
        from scripts import responses
        from iblnm import vis
        dropone_fn, total_r2_fn = responses._PERSESSION_DISPLAY_FNS[display]
        assert dropone_fn is getattr(vis, dropone_name)
        assert total_r2_fn is getattr(vis, total_r2_name)

    def test_invokes_mapped_pair_and_threads_pvalues(self, tmp_path):
        """Each mode calls the pair from the dispatch table; the per-mouse
        p-value table reaches the drop-one call only in ``session`` mode
        (subject/violin take none)."""
        import matplotlib.pyplot as plt
        from scripts import responses
        results = self._stub_frame()
        mouse_pvalues = pd.DataFrame({'subject': ['s0'], 'p_value': [0.01]})

        fig_dir = tmp_path / 'persession'
        fig_dir.mkdir()
        mocks = {mode: (MagicMock(return_value=plt.figure()),
                        MagicMock(return_value=plt.figure()))
                 for mode in ('session', 'subject', 'target')}
        with ExitStack() as stack:
            stack.enter_context(
                patch.dict(responses._PERSESSION_DISPLAY_FNS, mocks))
            for mode in ('session', 'subject', 'target'):
                responses.plot_persession_figures(results, mouse_pvalues,
                                                  fig_dir, display=mode)

        for mode, (dropone_mock, total_r2_mock) in mocks.items():
            dropone_mock.assert_called_once()
            total_r2_mock.assert_called_once()
        session_kwargs = mocks['session'][0].call_args.kwargs
        assert session_kwargs['mouse_pvalues'] is mouse_pvalues
        for mode in ('subject', 'target'):
            assert 'mouse_pvalues' not in mocks[mode][0].call_args.kwargs


class TestReadResultFrames:
    """The no-flag branch's read: written parquet in, narrowed frames out,
    with neither fitting pass touched."""

    _ROWS = [('eid-0', 'subj-0', 'VTA-r', 'r', 'VTA-DA'),
             ('eid-1', 'subj-1', 'DR-l', 'l', 'DR-5HT')]

    def _group(self, tmp_path):
        """Real group over two recordings, filtered to the first session."""
        from tests.test_data import _bare_group
        group = _bare_group(self._ROWS, tmp_path)
        group.filter_sessions(session_types=False, exclude_subjects=False,
                              exclude_eids=('eid-1',), qc_blockers=False,
                              targetnms=False, photometry_qc=False,
                              min_performance=False, required_contrasts=False)
        return group

    @staticmethod
    def _write(tmp_path):
        """The two grains the branch reads: eid-keyed, and cell-keyed."""
        keyed = pd.DataFrame({'eid': ['eid-0', 'eid-1'],
                              'brain_region': ['VTA-r', 'DR-l'],
                              'value': [1.0, 2.0]})
        unkeyed = pd.DataFrame({'target_NM': ['VTA-DA', 'DR-5HT'],
                                'regressor': ['contrast', 'contrast'],
                                'mean': [0.3, 0.4]})
        paths = {}
        for name, frame in (('ols', keyed), ('varcomp_summary', unkeyed)):
            paths[name] = tmp_path / f'{name}.parquet'
            frame.to_parquet(paths[name], index=False)
        return paths

    def test_narrows_keyed_frames_and_passes_the_others_whole(self, tmp_path):
        """A file covering a session the group's filters dropped is narrowed on
        load; the varcomp table, keyed by cell with no ``eid``, keeps every
        row rather than matching nothing."""
        from scripts.responses import read_result_frames
        frames = read_result_frames(self._group(tmp_path),
                                    self._write(tmp_path))

        assert list(frames['ols']['eid']) == ['eid-0']
        assert len(frames['varcomp_summary']) == 2

    def test_neither_pass_runs(self, tmp_path):
        """Both passes go through ``process``; reading the cached files runs
        neither."""
        from iblnm.data import PhotometrySessionGroup
        from scripts.responses import read_result_frames
        group = self._group(tmp_path)
        paths = self._write(tmp_path)

        with patch.object(PhotometrySessionGroup, 'process') as process:
            read_result_frames(group, paths)

        process.assert_not_called()


class TestTwoPassRun:
    """Both passes over a two-session store, and the files they write.

    The passes run for real — `group.process` over a written store, the
    script's own link functions — because what the wiring has to get right is
    the shape of what comes back: one trial-level frame and one fit frame per
    session, collected into the three output files at their own grains.
    """

    @staticmethod
    def _run(tmp_path):
        from tests.test_data import _persession_group
        from scripts.responses import build_donor_frame, fit_session
        from iblnm.config import LMM_FORMULAS
        group = _persession_group(
            tmp_path, [('eid-0', 'subj-0', 'VTA-r', 'r', 'VTA-DA'),
                       ('eid-1', 'subj-1', 'DR-l', 'l', 'DR-5HT')])

        donors = group.collect_donor_frames(group.process(build_donor_frame))
        returns = [pair for pair in
                   group.process(fit_session,
                                 formulas=LMM_FORMULAS['persession'],
                                 donors=donors)
                   if pair is not None]
        magnitudes = pd.concat([pair[0] for pair in returns],
                               ignore_index=True)
        ols, mouse = group.collect_fits([pair[1] for pair in returns])
        return donors, magnitudes, ols, mouse

    def test_donor_pool_holds_every_session(self, tmp_path):
        """Pass 1 emits no trial-level output and admits every session,
        including ones that produce no scorable fit."""
        donors, _, _, _ = self._run(tmp_path)
        assert list(donors) == ['eid-0', 'eid-1']

    def test_written_files_carry_the_schema_column_sets(self, tmp_path):
        """Round-tripped through parquet, each file holds its schema's columns
        at its own grain."""
        from iblnm.config import (OLS_PERSESSION_COLUMNS,
                                  RESPONSE_MAGNITUDE_COLUMNS)
        _, magnitudes, ols, mouse = self._run(tmp_path)

        paths = {}
        for name, frame in (('magnitudes', magnitudes[
                                RESPONSE_MAGNITUDE_COLUMNS]),
                            ('ols', ols), ('mouse', mouse)):
            paths[name] = tmp_path / f'{name}.parquet'
            frame.to_parquet(paths[name], index=False)
        read = {name: pd.read_parquet(path) for name, path in paths.items()}

        assert list(read['magnitudes'].columns) == RESPONSE_MAGNITUDE_COLUMNS
        assert list(read['ols'].columns) == OLS_PERSESSION_COLUMNS
        # Recording x event x trial, and recording x event x predictor.
        assert not read['magnitudes'].duplicated(
            subset=['eid', 'brain_region', 'event', 'trial']).any()
        assert not read['ols'].duplicated(
            subset=['eid', 'brain_region', 'event', 'predictor']).any()
        assert not read['mouse'].duplicated(
            subset=['target_NM', 'event', 'predictor', 'subject']).any()

    def test_ols_carries_the_whole_null_vector(self, tmp_path):
        """`null` survives the parquet round trip as an array of its own,
        which is what makes the per-mouse pooling recomputable without
        refitting — parquet stores it as a list column."""
        from iblnm.config import PERSESSION_PVAL_N_BOOTSTRAP
        _, _, ols, _ = self._run(tmp_path)
        path = tmp_path / 'ols_persession.parquet'
        ols.to_parquet(path, index=False)
        read = pd.read_parquet(path)

        vector = read['null'].iloc[0]
        assert isinstance(vector, np.ndarray)
        assert len(vector) == PERSESSION_PVAL_N_BOOTSTRAP


class TestReprocessWiring:
    """Source-level wiring of the two passes into the ``--reprocess`` branch."""

    def test_both_passes_run_through_process(self):
        reprocess, _ = _reprocess_and_default_branches()
        assert 'group.process(build_donor_frame)' in reprocess
        assert 'group.process(fit_session,' in reprocess
        assert 'collect_donor_frames(' in reprocess
        assert 'collect_fits(' in reprocess

    def test_neither_pass_is_parallelized(self):
        """Both passes stay sequential: `_process_parallel` pickles kwargs once
        per session, so a parallel pass would serialize the donor pool once for
        every session in it. No CLI flag offers otherwise."""
        src, _ = _responses_source()
        assert 'workers=' not in src.replace('`workers=1`', '')
        assert '--workers' not in src

    def test_the_three_files_are_written_from_the_collected_frames(self):
        reprocess, _ = _reprocess_and_default_branches()
        for name in ('magnitudes.to_parquet(', 'ols.to_parquet(',
                     'ols_mouse.to_parquet('):
            assert name in reprocess

    def test_empty_magnitudes_exits_nonzero(self):
        """The guard the collection pass inherited: nothing extracted means the
        store is not there to read."""
        reprocess, _ = _reprocess_and_default_branches()
        assert 'Check H5 files exist' in reprocess
        assert 'raise SystemExit(1)' in reprocess


def _responses_source():
    """Return (full source, __main__ block) of scripts/responses.py."""
    from pathlib import Path
    import scripts.responses
    src = Path(scripts.responses.__file__).read_text()
    main_block = src.split("if __name__ == '__main__':", 1)[1]
    return src, main_block


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
        # Its coefficients come out of the merged per-recording OLS frame the
        # collection returned, not a coefficients frame of its own.
        assert 'varcomp_coefficients(ols)' in reprocess
        assert 'RESPONSE_VARCOMP_SUMMARY_FPATH' in reprocess
        assert 'RESPONSE_VARCOMP_VIOLIN_FPATH' in reprocess
        assert reprocess.count('.to_parquet(') >= 2

    def test_default_reads_the_cached_varcomp_files(self):
        """The no-flag branch reads every cached frame through one call, the
        varcomp tables among them, rather than a loader method per file."""
        _, default = _reprocess_and_default_branches()
        assert 'read_result_frames(group)' in default
        assert "frames['varcomp_violin']" in default

    def test_violin_figure_plotted(self):
        _, main_block = _responses_source()
        assert 'plot_varcomp_violins(' in main_block
        assert 'response_varcomp_violins.svg' in main_block
