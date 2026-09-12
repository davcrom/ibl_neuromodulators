"""Tests for scripts/responses.py movement-encoding wiring."""
from contextlib import ExitStack
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from iblnm.config import (RESPONSE_DROPPED_TERMS, RESPONSE_MAGNITUDE_COLUMNS,
                          RESPONSES, STIM_ONSET_EVENT)
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
            'side': 'contra',
        })

    def test_uncorrected_means_are_the_raw_trace_means(self):
        """With the correction off, each condition's mean at each time point is
        the plain trial mean of the traces as stored."""
        from scripts.responses import condition_traces
        rec, ps = self._recording([[0., 1., 2., 3., 4.],
                                   [2., 3., 4., 5., 6.]])
        agg = condition_traces([(rec, ps)], self._trials(2),
                               ['feedback_times'], mode='pool', correct=False)

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
        agg = condition_traces([(rec, ps)], self._trials(2), ['feedback_times'],
                               mode='pool', correct=True)

        assert agg['mean'].tolist() == [-1., 0., 1., 2., 3.]
        assert agg['n'].tolist() == [2, 2, 2, 2, 1]

    def test_the_runs_masking_chronology_is_the_traces(self):
        """The masking events come from the run's ``config.RESPONSES`` entry,
        so a window with none — the pre-stimulus ``baseline`` — averages every
        sample the cut holds, feedback included."""
        from scripts.responses import condition_traces
        rec, ps = self._recording([[0., 1., 2., 3., 4.],
                                   [2., 3., 4., 5., 6.]],
                                  feedback_lags=[0.15, 10.0])
        agg = condition_traces([(rec, ps)], self._trials(2), [],
                               mode='pool', correct=True)

        assert agg['n'].tolist() == [2, 2, 2, 2, 2]

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

        agg = condition_traces(recordings, trials, ['feedback_times'],
                               mode=mode, correct=False)

        assert agg['mean'].tolist() == pytest.approx([expected] * 5)
        assert agg['n'].tolist() == [n] * 5

    def test_the_stored_averages_do_not_depend_on_the_mode(self):
        """Each session's per-condition averages are the same table whichever
        average is asked for, so one pass over the store answers both: only
        how they are combined afterwards differs."""
        import scripts.responses as responses
        from scripts.responses import condition_traces
        recordings, trials = self._uneven_cohort()

        stored = []
        for mode in ('pool', 'subject', 'subject_centered'):
            with patch.object(responses, 'aggregate_conditions') as reduce_:
                condition_traces(recordings, trials, ['feedback_times'],
                                 mode=mode, correct=False)
            stored.append(reduce_.call_args.args[0])

        for other in stored[1:]:
            pd.testing.assert_frame_equal(stored[0], other)

    def test_reduces_each_recording_before_reading_the_next(self):
        """Peak memory is what bounds this pass, so no per-trial sample
        survives the recording that produced it: the frame handed to the
        reduction carries one row per condition per recording — here 3
        recordings x 5 time points — not one per trial x time point."""
        import scripts.responses as responses
        from scripts.responses import condition_traces
        recordings, trials = self._uneven_cohort()

        with patch.object(responses, 'aggregate_conditions') as reduce_:
            condition_traces(recordings, trials, ['feedback_times'],
                             mode='subject_centered', correct=False)

        cells = reduce_.call_args.args[0]
        assert len(cells) == 15

    def test_averages_the_trials_the_models_are_fitted_on(self):
        """The frame handed in is the trial set, with no selection re-derived:
        the stored magnitudes already carry the trials the models fitted, so
        every row of them is averaged."""
        from scripts.responses import condition_traces
        rec, ps = self._recording([[0.] * 5, [10.] * 5, [100.] * 5])
        trials = self._trials(3)
        trials['probabilityLeft'] = [0.5, 0.8, 0.5]
        trials['choice'] = [1, 1, 0]

        agg = condition_traces([(rec, ps)], trials, ['feedback_times'],
                               mode='pool', correct=False)

        assert agg['mean'].tolist() == [pytest.approx(110 / 3)] * 5
        assert agg['n'].tolist() == [3] * 5

    def test_averages_a_frame_carrying_only_the_stored_columns(self):
        """The magnitude frame the script hands in carries
        ``RESPONSE_MAGNITUDE_COLUMNS`` plus whichever factors the run's
        ``RESPONSES`` entry derived, so the conditions averaged within must be
        drawn from that set alone."""
        from scripts.responses import condition_traces
        rec, ps = self._recording([[0., 1., 2., 3., 4.],
                                   [2., 3., 4., 5., 6.]])
        trials = self._trials(2)
        stored = trials[[col for col in RESPONSE_MAGNITUDE_COLUMNS
                         if col in trials]]

        agg = condition_traces([(rec, ps)], stored, ['feedback_times'],
                               mode='pool', correct=False)

        assert agg['mean'].tolist() == [1., 2., 3., 4., 5.]
        assert agg['n'].tolist() == [2] * 5


class _LinkSession:
    """Session stub exposing what the fitting link function calls and reads.

    The link function is the only place a session's state becomes group data,
    so what it is responsible for is sequencing the calls and handing back the
    right views — not the modelling itself, which is covered against real
    sessions in ``tests/test_data.py``. This stub records the arguments it was
    passed and plants the frames those steps would have produced, including
    the one piece of state the ordering turns on: ``filter_trials`` narrows
    the ``trials`` and ``response_magnitudes`` views, and trial 2 is the row it
    drops.
    """

    _MASKED_AWAY_TRIAL = 2

    def __init__(self):
        self._magnitudes = pd.DataFrame(
            {'trial': [0, 1, 2], 'response': [0.5, 0.7, np.nan],
             'hemisphere': ['r', 'r', 'r']})
        self._trials = pd.DataFrame(
            {'trial': [0, 1, 2], 'signed_contrast': [-25.0, 25.0, 100.0],
             'stim_side': ['left', 'right', 'right'], 'choice': [1, -1, 1],
             'reaction_time': [0.15, 0.30, 0.45]})
        self.trials = self._trials
        self.wheel_peak_velocity = np.array([1.0, 2.0, 3.0])
        self.fits = pd.DataFrame({'predictor': ['contrast'],
                                  'delta_r2': [0.04]})
        self.fitted = None
        self.filtered = None
        self.called = []

    def __getattr__(self, name):
        """Record the load and extract calls and do nothing else.

        The measurement sequence is what the link function is responsible for
        ordering, not for computing; the stub plants the frames those steps
        would have produced.
        """
        if name.startswith(('load_', 'extract_')):
            return lambda *args, **kwargs: self.called.append(
                (name, args, kwargs))
        raise AttributeError(name)

    def add_trial_columns(self, frame, name='peak_velocity'):
        """Assign by position, the real method's ndarray branch.

        Unlike the load and extract steps, this one leaves something the fit
        reads back — the reaction-time bin — so the stub does the work rather
        than only recording the call.
        """
        self.called.append(('add_trial_columns', (frame,), {}))
        self.trials[name] = frame
        return self.trials

    @property
    def response_magnitudes(self):
        return self._magnitudes[self._magnitudes['trial']
                                .isin(self.trials['trial'])]

    def masking_diagnostics(self):
        return self._magnitudes

    def fit_responses(self, formula, dropped_terms, donors, **criteria):
        self.fitted = (formula, dropped_terms, donors, criteria)
        return self.fits

    def filter_trials(self, **criteria):
        self.filtered = criteria
        self.trials = self._trials[self._trials['trial']
                                   != self._MASKED_AWAY_TRIAL]


class TestLinkFunctions:
    """The module-level functions ``group.process`` maps over the sessions:
    each sequences one session's own calls and returns plain data, holding no
    reference to a group."""

    def test_fit_session_forwards_its_arguments_and_returns_three_frames(self):
        from scripts.responses import PERSESSION_TRIAL_CRITERIA, fit_session
        ps = _LinkSession()
        formula = '{response} ~ contrast'
        dropped_terms = {'contrast': ['contrast']}
        donors = {'eid-1': 'donor'}

        magnitudes, fits, unfiltered = fit_session(
            ps, RESPONSES['stimulus'], formula, dropped_terms, donors)

        assert ps.fitted == (formula, dropped_terms, donors,
                             PERSESSION_TRIAL_CRITERIA)
        assert fits is ps.fits
        # The fitting loop leaves the mask on one fiber x event; the same
        # criteria are re-applied without one before the view is read.
        assert ps.filtered == PERSESSION_TRIAL_CRITERIA
        assert list(magnitudes['trial']) == [0, 1]
        # Right-hemisphere fiber, so the left stimulus of trial 0 is contra.
        assert list(magnitudes['side']) == ['contra', 'ipsi']

    def test_fit_session_returns_the_masking_frame_unfiltered(self):
        """The third frame is taken before the mask exists, so it keeps the
        trial the filtered view drops — the trial whose window was masked end
        to end is exactly what the masking diagnostic counts."""
        from scripts.responses import fit_session
        ps = _LinkSession()

        magnitudes, _, unfiltered = fit_session(
            ps, RESPONSES['stimulus'], '{response} ~ contrast',
            {'contrast': ['contrast']}, {})

        assert list(unfiltered['trial']) == [0, 1, 2]
        assert ps._MASKED_AWAY_TRIAL not in set(magnitudes['trial'])
        # Trial-level columns are joined on either side of the mask.
        assert 'side' in unfiltered.columns

    @pytest.mark.parametrize('window', list(RESPONSES))
    def test_the_entry_defines_the_measurement(self, window):
        """The run's entry — not a module-level default — supplies the window
        averaged, the events masked past, whether the baseline is subtracted
        and the one event measured, so a `baseline` run measures the
        pre-stimulus interval and a run carries a single event throughout."""
        from scripts.responses import fit_session
        ps = _LinkSession()
        entry = RESPONSES[window]

        fit_session(ps, entry, '{response} ~ contrast',
                    {'contrast': ['contrast']}, {})

        measured = [(args, kwargs) for name, args, kwargs in ps.called
                    if name == 'extract_response_magnitudes']
        assert measured == [((entry['window'], entry['masking_events'],
                              entry['baseline_correct']),
                             {'events': [entry['event']]})]


class TestReactionTimeBins:
    """``reaction_time_bin`` is the session's own reaction-time terciles, cut
    onto the trials table by the fitting pass."""

    @staticmethod
    def _bin(reaction_times):
        """The binning as ``fit_session`` applies it, for a bare array."""
        from scripts.responses import TERCILE_LABELS
        return pd.qcut(pd.Series(reaction_times), 3,
                       labels=TERCILE_LABELS).astype(str)

    def test_fit_session_bins_before_fitting(self):
        from scripts.responses import fit_session
        ps = _LinkSession()
        entry = {**RESPONSES['stimulus'],
                 'ANOVA': {'side': [], 'reaction_time_bin': []}}

        magnitudes, _, _ = fit_session(ps, entry, '{response} ~ contrast',
                                       {'contrast': ['contrast']}, {})

        assert set(magnitudes['reaction_time_bin']) <= {'low', 'mid', 'high'}

    def test_bins_are_ordered_terciles_of_the_session(self):
        """Three near-equal bins, ordered low to high by reaction time."""
        rng = np.random.default_rng(0)
        reaction_times = rng.uniform(0.1, 0.5, 30)

        binned = self._bin(reaction_times)

        counts = binned.value_counts()
        assert len(counts) == 3
        assert counts.max() - counts.min() <= 2
        means = pd.Series(reaction_times).groupby(binned).mean()
        assert means['low'] < means['mid'] < means['high']

    def test_binned_factor_keeps_every_subject_complete(self):
        """Each session yields all three bins, so no subject loses a cell."""
        from tests.test_data import _make_group_with_events
        group, magnitudes = _make_group_with_events()
        binned = pd.concat(
            [session.assign(
                reaction_time_bin=self._bin(session['reaction_time']).values)
             for _, session in magnitudes.groupby('eid')], ignore_index=True)

        result = group.response_anovaRM_fit(
            binned, {'reaction_time_bin': [], 'side': []},
            min_trials=5, min_subjects=2)

        assert len(result) > 0
        for table in result.values():
            assert (table['n_subjects'] == 3).all()
            assert (table['n_subjects_dropped'] == 0).all()


class TestPrepareDonor:
    """The first pass's link function, against real sessions.

    A donor frame is the response-independent selection coded on one frame per
    session, so what has to hold is the row set, the columns and the trial
    order — not the modelling, which never sees a donor's responses because it
    has none.
    """

    def test_donor_frame_is_built_without_loading_photometry(self, monkeypatch):
        from iblnm.config import PERSESSION_REGRESSORS
        from scripts.responses import prepare_donor
        from tests.test_data import _donorless_session

        ps = _donorless_session(unusable_trials=True)
        monkeypatch.setattr(
            PhotometrySession, 'load_responses',
            lambda *args, **kwargs: pytest.fail('photometry was loaded'))

        donor = prepare_donor(ps)
        assert set(PERSESSION_REGRESSORS) <= set(donor.frame.columns)
        assert 'response' not in donor.frame.columns
        assert donor.frame[PERSESSION_REGRESSORS].notna().all().all()

    def test_donor_rows_are_the_response_independent_selection(self):
        """Only the trials-only exclusions bite, so a donor frame is longer
        than the same session's focal cells, which also drop the trials whose
        response is null."""
        from iblnm.data import response_column
        from scripts.responses import (PERSESSION_TRIAL_CRITERIA,
                                       prepare_donor)
        from tests.test_data import (_add_second_recording, _donorless_session,
                                     _make_session_for_persession,
                                     _measured_session)

        # The second recording carries no signal on its first 30 trials, so
        # its cells lose those rows and the donor frame does not.
        ps = _measured_session(_add_second_recording(
            _make_session_for_persession(unusable_trials=True), n_missing=30))
        trials = ps.trials
        expected = ((trials['choice'] != 0)
                    & (trials['response_times']
                       - trials['stimOnTrigger_times'] > 0.05)
                    & (trials['firstMovement_times']
                       - trials['stimOnTrigger_times'] > 0)
                    & trials[['contrast', 'stim_side',
                              'feedbackType']].notna().all(axis=1)
                    & ~np.isnan(ps.wheel_peak_velocity)).sum()

        criteria = dict(PERSESSION_TRIAL_CRITERIA)
        ps.add_trial_columns(ps.cell_magnitudes('DR-l', STIM_ONSET_EVENT))
        ps.filter_trials(
            **{**criteria,
               'complete': [*criteria['complete'],
                            response_column('DR-l', STIM_ONSET_EVENT)]})
        donor = prepare_donor(_donorless_session(unusable_trials=True))

        assert len(donor.frame) == expected
        assert len(donor.frame) > len(ps.trials)

    def test_donor_frame_drops_the_trials_coding_would_falsify(self):
        """A blank the coding hides is still a hole.

        `code_predictors` codes `side`, `choice_side` and `reward` through
        `np.where`, so a blank `stim_side` or `feedbackType` comes out as the
        opposite level rather than as a blank — nothing downstream could catch
        it. The five defective trials of the fixture are the criteria's whole
        job: no log, no peak velocity, and the three blanks.
        """
        from scripts.responses import prepare_donor
        from tests.test_data import _donorless_session

        donor = prepare_donor(_donorless_session(unusable_trials=True))

        assert set(donor.frame['trial']).isdisjoint(range(5))

    def test_donor_frame_preserves_trial_order(self):
        from scripts.responses import prepare_donor
        from tests.test_data import _donorless_session

        ps = _donorless_session()
        every_trial = set(ps.trials['trial'])
        trial = prepare_donor(ps).frame['trial']
        assert trial.is_monotonic_increasing
        assert set(trial) <= every_trial


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

    def test_counts_the_trials_no_model_saw(self):
        """The frame is counted as handed in, not re-selected.

        A no-go trial whose window was masked end to end carries no response
        and is fitted by nothing; it is exactly what the diagnostic exists to
        report, so it must survive into the count. Re-deriving a modeling
        selection here would drop it and report no masking at all.
        """
        from scripts.responses import compute_masking_diagnostics
        magnitudes = self._magnitudes([1.0, 0.0])
        magnitudes['choice'] = [0, 1]
        magnitudes['response'] = [np.nan, 1.0]

        cell = compute_masking_diagnostics(
            magnitudes, window=self._WINDOW).iloc[0]

        assert cell['n_trials'] == 2
        assert cell['pct_fully_masked'] == pytest.approx(50.0)

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
                self._group(), pd.DataFrame(), ['feedback_times'], tmp_path)

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
                self._group(), pd.DataFrame(), ['feedback_times'], tmp_path)

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
        trials = magnitudes

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
        """Per-recording OLS rows over all eighteen drop-one labels.

        Main-effect ΔR² runs 0.10-0.20 and interaction ΔR² 0.001-0.002, the
        order-of-magnitude gap the two shared y-scales exist for.
        """
        rows = []
        for s in range(3):
            for p in RESPONSE_DROPPED_TERMS:
                delta = (0.001 if ':' in p else 0.1) * (1 + s / 2)
                rows.append({
                    'eid': f'e{s}', 'subject': f's{s}', 'target_NM': 'VTA-DA',
                    'brain_region': 'VTA', 'event': 'stimOnTrigger_times',
                    'predictor': p, 'n_trials': 100, 'r2_full': 0.5,
                    'r2_full_adj': 0.45, 'delta_r2': delta + 0.02,
                    'delta_r2_adj': delta,
                    'coef': 0.2, 'coef_se': 0.05, 'p_value': 0.01,
                    'q_value': 0.02, 'n_donors': 700})
        return pd.DataFrame(rows)

    def _two_target_frame(self):
        """The stub frame for VTA-DA plus a DR-5HT copy with its own mice and
        three times the ΔR², so the two targets span different ranges."""
        frame = self._stub_frame()
        return pd.concat([frame, frame.assign(
            target_NM='DR-5HT', brain_region='DR',
            subject='d' + frame['subject'], eid='d' + frame['eid'],
            delta_r2_adj=3 * frame['delta_r2_adj'])], ignore_index=True)

    def test_ylim_padded_around_the_class_values(self):
        """``dropone_ylim`` spans the class's values with a 5% pad, and has no
        range to give when the values are absent or all equal."""
        from scripts.responses import dropone_ylim
        from iblnm.vis import DROPONE_TERM_CLASSES
        frame = self._stub_frame()

        low, high = dropone_ylim(frame, DROPONE_TERM_CLASSES['main'])
        assert (low, high) == pytest.approx((0.1 - 0.005, 0.2 + 0.005))
        flat = frame.assign(delta_r2_adj=0.03)
        assert dropone_ylim(flat, ['contrast']) is None  # one value repeated
        assert dropone_ylim(frame, ['no_such_term']) is None

    def test_one_file_per_dropped_term_and_per_target_nm(self, tmp_path):
        """The frame and the output directory are the whole input — no group.
        One drop-one figure per label, named for it with the interaction colon
        replaced, one per target-NM named for it, plus the single full-model R²
        figure, all in one directory. The per-mouse table is absent here, so
        this exercises the real figures rather than a mock."""
        from scripts import responses

        fig_dir = tmp_path / 'persession'
        fig_dir.mkdir()
        responses.plot_persession_figures(self._two_target_frame(), None,
                                          fig_dir)

        names = {p.name for p in fig_dir.glob('*.svg')}
        assert len(names) == len(RESPONSE_DROPPED_TERMS) + 2 + 1
        assert all(':' not in name for name in names)
        assert {'contrast.svg', 'contrast-side.svg'} <= names
        assert {'VTA-DA.svg', 'DR-5HT.svg'} <= names
        assert all(p.stat().st_size > 0 for p in fig_dir.glob('*.svg'))

    @pytest.mark.parametrize(
        'display, dropone_name, target_name, total_r2_name', [
        ('session', 'plot_ols_dropone', 'plot_ols_dropone_target',
         'plot_ols_total_r2'),
        ('subject', 'plot_ols_dropone_subject',
         'plot_ols_dropone_target_subject', 'plot_ols_total_r2_subject'),
        ('target', 'plot_ols_dropone_violin', 'plot_ols_dropone_target_violin',
         'plot_ols_total_r2_violin'),
    ])
    def test_display_maps_to_function_triple(self, display, dropone_name,
                                             target_name, total_r2_name):
        """The dispatch table gives each display mode its matching (per-term
        drop-one, per-target drop-one, full-model R²) vis functions."""
        from scripts import responses
        from iblnm import vis
        dropone_fn, target_fn, total_r2_fn = (
            responses._PERSESSION_DISPLAY_FNS[display])
        assert dropone_fn is getattr(vis, dropone_name)
        assert target_fn is getattr(vis, target_name)
        assert total_r2_fn is getattr(vis, total_r2_name)

    def test_invokes_mapped_functions_and_threads_pvalues(self, tmp_path):
        """Each mode calls its dispatch-table functions — the per-term drop-one
        function once per dropped term, the per-target one once per target-NM,
        the full-model R² function once; the per-mouse p-value table reaches
        the drop-one calls only in ``session`` mode (subject/violin take
        none)."""
        import matplotlib.pyplot as plt
        from scripts import responses
        results = self._two_target_frame()
        mouse_pvalues = pd.DataFrame({'subject': ['s0'], 'p_value': [0.01]})

        fig_dir = tmp_path / 'persession'
        fig_dir.mkdir()
        mocks = {mode: tuple(MagicMock(return_value=plt.figure())
                             for _ in range(3))
                 for mode in ('session', 'subject', 'target')}
        with ExitStack() as stack:
            stack.enter_context(
                patch.dict(responses._PERSESSION_DISPLAY_FNS, mocks))
            for mode in ('session', 'subject', 'target'):
                responses.plot_persession_figures(results, mouse_pvalues,
                                                  fig_dir, display=mode)

        for mode, (dropone_mock, target_mock, total_r2_mock) in mocks.items():
            assert dropone_mock.call_count == len(RESPONSE_DROPPED_TERMS)
            assert target_mock.call_count == 2
            total_r2_mock.assert_called_once()
        for fn_mock in mocks['session'][:2]:
            assert fn_mock.call_args.kwargs['mouse_pvalues'] is mouse_pvalues
        for mode in ('subject', 'target'):
            assert all('mouse_pvalues' not in fn_mock.call_args.kwargs
                       for fn_mock in mocks[mode][:2])

    def test_ylim_shared_within_term_class_and_differs_between(self, tmp_path):
        """Every main-effect figure carries one range and every interaction
        figure another, so a term near zero renders on its class's range
        rather than autoscaled to itself."""
        import matplotlib.pyplot as plt
        from scripts import responses
        from iblnm.vis import DROPONE_TERM_CLASSES

        fig_dir = tmp_path / 'persession'
        fig_dir.mkdir()
        # One interaction contributes nothing, so its figure's range can only
        # come from the class rather than from its own values.
        results = self._stub_frame()
        flat = results['predictor'] == 'contrast:side'
        results.loc[flat, 'delta_r2_adj'] = 0.0
        dropone_mock = MagicMock(return_value=plt.figure())
        with patch.dict(responses._PERSESSION_DISPLAY_FNS,
                        {'session': (dropone_mock,
                                     MagicMock(return_value=plt.figure()),
                                     MagicMock(return_value=plt.figure()))}):
            responses.plot_persession_figures(results, None, fig_dir)

        ylim_by_term = {call.kwargs['predictor']: call.kwargs['ylim']
                        for call in dropone_mock.call_args_list}
        mains = {ylim_by_term[term] for term in DROPONE_TERM_CLASSES['main']}
        interactions = {ylim_by_term[term]
                        for term in DROPONE_TERM_CLASSES['interaction']}
        assert len(mains) == 1 and len(interactions) == 1
        assert mains != interactions
        # The stub's mains span 0.10-0.20, the interactions 0.001-0.002: each
        # class's range covers its own values and not the other's.
        (main_low, main_high), = mains
        (int_low, int_high), = interactions
        assert main_low < 0.1 and main_high > 0.2
        assert int_low < 0.001 and int_high > 0.002 and int_high < 0.1

    def test_ylim_per_target_nm_from_its_own_rows(self, tmp_path):
        """Each per-target figure spans its own target's main-effect ΔR², so
        two targets with different spreads get different ranges."""
        import matplotlib.pyplot as plt
        from scripts import responses
        from iblnm.vis import DROPONE_TERM_CLASSES

        fig_dir = tmp_path / 'persession'
        fig_dir.mkdir()
        results = self._two_target_frame()
        target_mock = MagicMock(return_value=plt.figure())
        with patch.dict(responses._PERSESSION_DISPLAY_FNS,
                        {'session': (MagicMock(return_value=plt.figure()),
                                     target_mock,
                                     MagicMock(return_value=plt.figure()))}):
            responses.plot_persession_figures(results, None, fig_dir)

        ylim_by_target = {call.kwargs['target_nm']: call.kwargs['ylim']
                          for call in target_mock.call_args_list}
        # VTA-DA's mains span 0.10-0.20, DR-5HT's three times that.
        mains = DROPONE_TERM_CLASSES['main']
        assert ylim_by_target['VTA-DA'] == pytest.approx(
            (0.1 - 0.005, 0.2 + 0.005))
        assert ylim_by_target['DR-5HT'] == pytest.approx(
            (0.3 - 0.015, 0.6 + 0.015))
        assert all(call.kwargs['terms'] == mains
                   for call in target_mock.call_args_list)


class TestWindowOutputs:
    """One window per run, one directory tree per window."""

    def test_directories_are_created_under_the_window(self, tmp_path,
                                                      monkeypatch):
        from scripts import responses
        monkeypatch.setattr(responses, 'RESPONSES_DIR', tmp_path / 'results')
        monkeypatch.setattr(responses, 'RESPONSE_FIGURES_DIR',
                            tmp_path / 'figures')

        data_dir, fig_dirs = responses.output_dirs('baseline')

        assert data_dir == tmp_path / 'results/baseline'
        assert fig_dirs['persession'] == (tmp_path
                                          / 'figures/baseline/persession')
        assert data_dir.is_dir()
        assert all(d.is_dir() for d in fig_dirs.values())

    def test_result_paths_sit_in_the_window_directory(self, tmp_path):
        """The frame names are relative, so the run's directory decides where
        they are written and read back."""
        from scripts.responses import RESULT_FPATHS, result_paths
        paths = result_paths(tmp_path / 'stimulus')

        assert set(paths) == set(RESULT_FPATHS)
        assert paths['magnitudes'] == (tmp_path / 'stimulus'
                                       / 'response_magnitudes.parquet')

    @pytest.mark.parametrize('window', list(RESPONSES))
    def test_run_config_round_trips(self, tmp_path, window):
        """The entry the run used is written beside its tables; the tuple
        window comes back as a list, and every other value unchanged."""
        import json
        from scripts.responses import write_run_config
        entry = RESPONSES[window]

        path = write_run_config(entry, tmp_path)
        read = json.loads(path.read_text())

        assert read['window'] == list(entry['window'])
        assert {k: v for k, v in read.items() if k != 'window'} == {
            k: v for k, v in entry.items() if k != 'window'}


class TestParseArgs:
    """The CLI: one ``config.RESPONSES`` window per invocation, named."""

    @pytest.mark.parametrize('window', list(RESPONSES))
    def test_each_window_is_accepted(self, window):
        from scripts.responses import parse_args
        args = parse_args([window])

        assert args.window == window

    def test_unknown_window_is_rejected(self):
        from scripts.responses import parse_args
        with pytest.raises(SystemExit):
            parse_args(['no_such_window'])

    def test_window_is_mandatory(self):
        """No window means no analysis unit, so the run cannot be inferred."""
        from scripts.responses import parse_args
        with pytest.raises(SystemExit):
            parse_args([])

    def test_the_other_flags_are_unchanged(self):
        from scripts.responses import parse_args
        args = parse_args(['feedback', '--reprocess',
                           '--persession-display', 'target'])

        assert (args.reprocess, args.persession_display) == (True, 'target')


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
        unkeyed = pd.DataFrame({'subject': ['subj-0', 'subj-1'],
                                'target_NM': ['VTA-DA', 'DR-5HT'],
                                'predictor': ['contrast', 'contrast'],
                                'p_value': [0.3, 0.4]})
        paths = {}
        for name, frame in (('ols', keyed), ('ols_mouse', unkeyed)):
            paths[name] = tmp_path / f'{name}.parquet'
            frame.to_parquet(paths[name], index=False)
        return paths

    def test_narrows_keyed_frames_and_passes_the_others_whole(self, tmp_path):
        """A file covering a session the group's filters dropped is narrowed on
        load; the per-mouse table, keyed by cell with no ``eid``, keeps every
        row rather than matching nothing."""
        from scripts.responses import read_result_frames
        frames = read_result_frames(self._group(tmp_path),
                                    self._write(tmp_path))

        assert list(frames['ols']['eid']) == ['eid-0']
        assert len(frames['ols_mouse']) == 2

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
    the shape of what comes back: three frames per session, collected into the
    output files at their own grains.
    """

    @staticmethod
    def _run(tmp_path):
        from tests.test_data import _persession_group, _response_model
        from scripts.responses import fit_session, prepare_donor
        group = _persession_group(
            tmp_path, [('eid-0', 'subj-0', 'VTA-r', 'r', 'VTA-DA'),
                       ('eid-1', 'subj-1', 'DR-l', 'l', 'DR-5HT')],
            unusable_trials=True)

        donors = group.collect_donor_frames(group.process(prepare_donor))
        formula, dropped_terms = _response_model()
        returns = [frames for frames in
                   group.process(fit_session, entry=RESPONSES['stimulus'],
                                 formula=formula,
                                 dropped_terms=dropped_terms, donors=donors)
                   if frames is not None]
        magnitudes = pd.concat([frames[0] for frames in returns],
                               ignore_index=True)
        ols, mouse = group.collect_fits([frames[1] for frames in returns])
        unfiltered = pd.concat([frames[2] for frames in returns],
                               ignore_index=True)
        return donors, magnitudes, ols, mouse, unfiltered

    def test_donor_pool_holds_every_session(self, tmp_path):
        """Pass 1 emits no trial-level output and admits every session,
        including ones that produce no scorable fit."""
        donors = self._run(tmp_path)[0]
        assert list(donors) == ['eid-0', 'eid-1']

    def test_written_magnitudes_are_the_fitted_trials(self, tmp_path):
        """The stored table carries the rows the models were fitted on: every
        fiber x event of a session covers the same trials, and that trial count
        is the ``n_trials`` its fits report."""
        _, magnitudes, ols, _, _ = self._run(tmp_path)

        for eid, rows in magnitudes.groupby('eid'):
            per_cell = rows.groupby(['brain_region', 'event'])['trial'].apply(
                frozenset)
            assert len(set(per_cell)) == 1
            assert set(ols.loc[ols['eid'] == eid, 'n_trials']) == {
                len(per_cell.iloc[0])}

    def test_written_files_carry_the_schema_column_sets(self, tmp_path):
        """Round-tripped through parquet, each file holds its schema's columns
        at its own grain."""
        from iblnm.config import (OLS_PERSESSION_COLUMNS,
                                  RESPONSE_MAGNITUDE_COLUMNS)
        _, magnitudes, ols, mouse, _ = self._run(tmp_path)

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
        refitting — parquet stores it as a list column. One entry per donor:
        the other session is this one's only admitted donor."""
        _, _, ols, _, _ = self._run(tmp_path)
        path = tmp_path / 'ols_persession.parquet'
        ols.to_parquet(path, index=False)
        read = pd.read_parquet(path)

        row = read.iloc[0]
        assert isinstance(row['null'], np.ndarray)
        assert row['n_donors'] == 1
        assert len(row['null']) == row['n_donors']


class TestReprocessWiring:
    """Source-level wiring of the two passes into the ``--reprocess`` branch."""

    def test_both_passes_run_through_process(self):
        reprocess, _ = _reprocess_and_default_branches()
        assert 'group.process(prepare_donor)' in reprocess
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
        assert 'No response magnitudes' in reprocess
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
