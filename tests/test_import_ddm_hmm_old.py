"""Tests for scripts/import_ddm_hmm_old.py — legacy DDM-HMM fits into the store."""
from unittest.mock import MagicMock

import h5py
import numpy as np
import pandas as pd
import pytest

import scripts.import_ddm_hmm_old as import_old
from iblnm.data import PhotometrySession


def _stored_trials():
    """Six stored trials; two (rows 1, 3) are absent from the fit.

    Kept trials 0, 2, 4, 5 have RTs 0.5, 0.3, 0.7, 0.2 s and |contrast|
    100, 12.5, 25, 6.25 %; the two dropped trials carry RTs (0.4, 10.5)
    that appear in no fit row, so the ordered RT alignment skips them.
    The `trial` identities are not the row positions, so a result keyed by
    position rather than identity fails. `signed_contrast` is in percent.
    """
    return pd.DataFrame({
        'trial':           [10,   11,    12,    13,    14,    15  ],
        'stimOn_times':    [0.0,  1.0,   2.0,   3.0,   4.0,   5.0 ],
        'response_times':  [0.5,  1.4,   2.3,   13.5,  4.7,   5.2 ],
        'signed_contrast': [100,  25.0,  -12.5, 6.25,  -25.0, 6.25],
    })


def _posterior_rows(eid='eid-a'):
    """Fit rows for the four kept trials, shuffled by `trial_in_dataset`.

    `trial_in_dataset` encodes chronological order. `signed_contrast` is a
    fraction whose sign may differ from the store's (collaborator coding).
    """
    return pd.DataFrame([
        {'eid': eid, 'trial_in_dataset': 2, 'rt': 0.7, 'signed_contrast': 0.25,
         'map_state': 2, 'state_1': 0.1, 'state_2': 0.9},
        {'eid': eid, 'trial_in_dataset': 0, 'rt': 0.5, 'signed_contrast': 1.0,
         'map_state': 2, 'state_1': 0.1, 'state_2': 0.9},
        {'eid': eid, 'trial_in_dataset': 3, 'rt': 0.2, 'signed_contrast': 0.0625,
         'map_state': 1, 'state_1': 0.6, 'state_2': 0.4},
        {'eid': eid, 'trial_in_dataset': 1, 'rt': 0.3, 'signed_contrast': -0.125,
         'map_state': 1, 'state_1': 0.6, 'state_2': 0.4},
    ])


class TestAlignSession:

    def test_aligns_kept_trials(self):
        aligned = import_old.align_session(_posterior_rows(), _stored_trials())

        assert list(aligned['trial']) == [10, 12, 14, 15]
        assert list(aligned['map_state']) == [2, 1, 2, 1]
        assert list(aligned['state_2']) == [0.9, 0.4, 0.9, 0.4]

    def test_rt_no_match_raises(self):
        rows = _posterior_rows()
        rows.loc[0, 'rt'] = 99.0  # no trial has this RT
        with pytest.raises(ValueError, match='RT match'):
            import_old.align_session(rows, _stored_trials())

    def test_contrast_mismatch_raises(self):
        rows = _posterior_rows()
        rows.loc[0, 'signed_contrast'] = 0.5  # 50 % vs stored 25 % on that trial
        with pytest.raises(ValueError, match='contrast'):
            import_old.align_session(rows, _stored_trials())


def _write_old_fit(old_dir, subject='ZFM-00001', k=2):
    """Synthetic old-format CSVs for one mouse, plus a K=3 decoy and a decoy mouse."""
    old_dir.mkdir(parents=True, exist_ok=True)
    states = range(1, k + 1)
    pd.DataFrame({
        'mouse': [subject] * k + ['ZFM-99999'] * k, 'best_K': k,
        'n_trials': 100, 'alpha': 0.2, 'state': [*states, *states],
        'B': 1.0, 'k': 2.0, 'a0': 0.5, 'tau': 0.0,
        'self_transition': [0.8, 0.7, 0.1, 0.1],
        'occupancy': [0.6, 0.4, 0.5, 0.5],
    }).to_csv(old_dir / 'all_mice_bestK_params.csv', index=False)
    pd.DataFrame({
        'subject': [subject, subject, 'ZFM-99999'], 'K': [k, k + 1, k],
        'n_trials': 100, 'n_params': [10, 18, 10],
        'logL': [-50.0, -40.0, -30.0], 'bic': [110.0, 100.0, 70.0],
    }).to_csv(old_dir / 'model_selection_all.csv', index=False)
    pd.DataFrame({
        'state': states, 'B': [4.0, 3.0], 'k': [0.1, 0.3], 'alpha': 0.2,
        'a0': [0.7, 0.2], 'tau': 0.0, 'pi0': [0.9, 0.1],
    }).to_csv(old_dir / f'{subject}_K{k}_params.csv', index=False)
    pd.DataFrame({
        'from_state': states, 'to_1': [0.8, 0.3], 'to_2': [0.2, 0.7],
    }).to_csv(old_dir / f'{subject}_K{k}_transition.csv', index=False)
    _posterior_rows().to_csv(old_dir / f'{subject}_K{k}_posteriors.csv',
                             index=False)


class TestReadFit:

    def test_attrs_carry_new_format_names(self, tmp_path):
        _write_old_fit(tmp_path)

        k, attrs = import_old.read_fit('ZFM-00001', tmp_path)

        assert k == 2
        assert not {'bic', 'logL', 'pi0', 'K', 'to_1'} & set(attrs)
        assert attrs['BIC'] == 110.0
        assert attrs['loglik'] == -50.0
        assert attrs['K_ddm'] == 2
        assert list(attrs['init']) == [0.9, 0.1]
        assert list(attrs['a0']) == [0.7, 0.2]
        assert list(attrs['trans_to_1']) == [0.8, 0.3]
        assert list(attrs['trans_to_2']) == [0.2, 0.7]
        assert list(attrs['self_transition']) == [0.8, 0.7]
        assert list(attrs['occupancy']) == [0.6, 0.4]
        assert np.isnan(attrs['converged'])
        assert np.isnan(attrs['restart'])


class TestImportSession:

    EID = 'eid-a'

    def _session(self, tmp_path, trials=True):
        """A session whose file holds (optionally) trials and a stale `hmm/ddm-k9`."""
        row = pd.Series({
            'eid': self.EID, 'subject': 'ZFM-00001',
            'start_time': '2024-01-01T10:00:00', 'number': 1,
            'lab': 'test_lab', 'projects': [], 'url': None, 'session_n': 1,
            'task_protocol': '_iblrig_tasks_biasedChoiceWorld6.4.2',
            'session_type': 'biased', 'datasets': [],
        })
        writer = PhotometrySession(row, one=MagicMock(), load_data=False)
        writer.filepath = tmp_path / f'{self.EID}.h5'
        if trials:
            writer.trials = _stored_trials()
        writer.hmm = {9: {'trials': pd.DataFrame({'trial': [10]}),
                          'attrs': {'BIC': 9.0}}}
        writer.save_h5(groups=['trials', 'hmm'] if trials else ['hmm'])

        ps = PhotometrySession(row, one=MagicMock(), load_data=False)
        ps.filepath = writer.filepath
        return ps

    def _stored_ks(self, fpath):
        with h5py.File(fpath, 'r') as h5:
            return sorted(h5['hmm']) if 'hmm' in h5 else []

    def test_writes_renamed_matched_rows(self, tmp_path):
        ps = self._session(tmp_path)
        posteriors = pd.concat([
            _posterior_rows(self.EID),
            # Another session's rows, whose RTs match none of these trials.
            _posterior_rows('other-eid').assign(rt=9.9),
        ], ignore_index=True)

        written = import_old.import_session(ps, 2, posteriors, {'BIC': 110.0})

        assert written == [2]
        assert self._stored_ks(ps.filepath) == ['ddm-k2']
        stored = ps.load_hmm(2)
        fit_trials = stored['trials']
        assert list(fit_trials['trial']) == [10, 12, 14, 15]
        assert list(fit_trials['p_state_2']) == [0.9, 0.4, 0.9, 0.4]
        assert list(fit_trials['map_state']) == [2, 1, 2, 1]
        assert not {'state_1', 'state_2', 'viterbi_state'} & set(fit_trials)
        assert stored['attrs']['BIC'] == 110.0

    def test_no_stored_trials_raises(self, tmp_path):
        ps = self._session(tmp_path, trials=False)
        with pytest.raises(ValueError, match='trials'):
            import_old.import_session(ps, 2, _posterior_rows(self.EID), {})
        assert self._stored_ks(ps.filepath) == ['ddm-k9']


class TestSelectSubjects:

    def _dirs(self, tmp_path):
        """Old fits for ZFM-A and ZFM-B; a new-format directory for ZFM-A."""
        old_dir, new_dir = tmp_path / 'old', tmp_path / 'new'
        _write_old_fit(old_dir, subject='ZFM-A')
        pd.DataFrame({'mouse': ['ZFM-A', 'ZFM-A', 'ZFM-B', 'ZFM-B']}).to_csv(
            old_dir / 'all_mice_bestK_params.csv', index=False)
        (new_dir / 'ZFM-A').mkdir(parents=True)
        return old_dir, new_dir

    def test_skips_subjects_with_new_format_fit(self, tmp_path):
        assert import_old.select_subjects(*self._dirs(tmp_path)) == ['ZFM-B']

    def test_named_subjects_still_skip_new_format(self, tmp_path):
        old_dir, new_dir = self._dirs(tmp_path)
        assert import_old.select_subjects(old_dir, new_dir,
                                          ['ZFM-A', 'ZFM-B']) == ['ZFM-B']
        assert import_old.select_subjects(old_dir, new_dir, ['ZFM-A']) == []
