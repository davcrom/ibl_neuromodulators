"""Tests for scripts/import_ddm_hmm.py — new-format DDM-HMM fits into the store."""
from unittest.mock import MagicMock

import h5py
import numpy as np
import pandas as pd
import pytest

import scripts.import_ddm_hmm as import_ddm_hmm
from iblnm.data import PhotometrySession


def _stored_trials():
    """The H5 trials columns the alignment reads: contrast in percent."""
    return pd.DataFrame({
        'trial': [0, 1, 2, 3],
        'contrast': [0.0, 12.5, 100.0, 25.0],
        'choice': [-1.0, 1.0, 0.0, 1.0],
    })


def _fit_trials(eid='eid-a'):
    """A fit's rows for the same trials: contrast as a fraction, choice flipped."""
    return pd.DataFrame({
        'eid': [eid] * 4,
        'trial_n': [0, 1, 2, 3],
        'c': [0.0, 0.125, 1.0, 0.25],
        'choice': [1, -1, 0, -1],
        'viterbi_state': [1, 2, 4, 1],
    })


class TestCheckAlignment:

    def test_matched_pair_passes(self):
        import_ddm_hmm.check_alignment(_fit_trials(), _stored_trials())

    def test_row_count_mismatch_raises(self):
        with pytest.raises(ValueError, match='trials'):
            import_ddm_hmm.check_alignment(_fit_trials().iloc[:3],
                                           _stored_trials())

    def test_trial_absent_from_store_raises(self):
        fit = _fit_trials()
        fit.loc[3, 'trial_n'] = 7
        with pytest.raises(ValueError, match='trial'):
            import_ddm_hmm.check_alignment(fit, _stored_trials())

    def test_contrast_mismatch_raises(self):
        fit = _fit_trials()
        fit.loc[1, 'c'] = 0.0625
        with pytest.raises(ValueError, match='contrast'):
            import_ddm_hmm.check_alignment(fit, _stored_trials())

    def test_choice_not_flipped_raises(self):
        fit = _fit_trials()
        fit.loc[0, 'choice'] = -1
        with pytest.raises(ValueError, match='choice'):
            import_ddm_hmm.check_alignment(fit, _stored_trials())


def _write_fit_files(subject_dir, k=3):
    """Synthetic `trials_K{k}.csv` and `params_K{k}.csv`, with Unicode names."""
    subject_dir.mkdir(parents=True, exist_ok=True)
    _fit_trials().to_csv(subject_dir / f'trials_K{k}.csv', index=False)
    n_states = k + 1
    pd.DataFrame({
        'state': range(1, n_states + 1),
        'kind': ['ddm'] * k + ['omission'],
        'B': [1.0] * k + [np.nan], 'k': [2.0] * k + [np.nan],
        'α': [.3] * k + [np.nan], 'a₀': [.5] * k + [np.nan],
        'τ': [.1] * k + [np.nan],
        'init': [1 / n_states] * n_states,
        **{f'trans_to_{j}': [1 / n_states] * n_states
           for j in range(1, n_states + 1)},
    }).to_csv(subject_dir / f'params_K{k}.csv', index=False)


def _comparison(subject='ZFM-00001', k=3):
    return pd.DataFrame({
        'subject': [subject, subject, 'ZFM-99999'], 'K_ddm': [k, k + 1, k],
        'BIC': [10.0, 20.0, 30.0], 'converged': [True, False, True],
        'host': ['scc-a', 'scc-b', 'scc-c'],
    })


class TestReadFits:

    def test_reads_each_k_with_ascii_params(self, tmp_path):
        subject_dir = tmp_path / 'ZFM-00001'
        _write_fit_files(subject_dir, k=3)

        fits = import_ddm_hmm.read_fits(subject_dir, _comparison())

        assert set(fits) == {3}
        attrs = fits[3]['attrs']
        assert not {'α', 'a₀', 'τ'} & set(attrs)
        for name in ('alpha', 'a0', 'tau'):
            assert isinstance(attrs[name], np.ndarray) and len(attrs[name]) == 4
        assert attrs['BIC'] == 10.0
        assert attrs['host'] == 'scc-a'
        assert list(attrs['kind']) == ['ddm'] * 3 + ['omission']
        assert attrs['kind'].dtype.kind == 'U'
        assert len(attrs['trans_to_4']) == 4
        pd.testing.assert_frame_equal(fits[3]['trials'], _fit_trials())


def _fit(eids, k=3):
    """One K as `read_fits` returns it, holding rows for each of `eids`."""
    return {'trials': pd.concat([_fit_trials(eid) for eid in eids],
                                ignore_index=True),
            'attrs': {'BIC': float(k), 'kind': np.array(['ddm'] * k + ['omission'])}}


class TestImportSession:

    EID = 'eid-a'

    def _session(self, tmp_path, trials=True):
        """A session whose file holds trials and a stale `hmm/ddm-k9`."""
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
        writer.hmm = {9: _fit([self.EID], k=9)}
        writer.save_h5(groups=['trials', 'hmm'] if trials else ['hmm'])

        ps = PhotometrySession(row, one=MagicMock(), load_data=False)
        ps.filepath = writer.filepath
        return ps

    def _stored_ks(self, fpath):
        with h5py.File(fpath, 'r') as h5:
            return sorted(h5['hmm']) if 'hmm' in h5 else []

    def test_writes_only_imported_ks(self, tmp_path):
        ps = self._session(tmp_path)
        fits = {3: _fit([self.EID, 'eid-b'], k=3),
                4: _fit([self.EID], k=4),
                5: _fit(['eid-b'], k=5)}

        assert import_ddm_hmm.import_session(ps, fits) == [3, 4]

        assert self._stored_ks(ps.filepath) == ['ddm-k3', 'ddm-k4']
        stored = ps.load_hmm(3)
        assert (stored['trials']['eid'] == self.EID).all()
        assert list(stored['trials']['trial']) == list(stored['trials']['trial_n'])
        assert stored['attrs']['BIC'] == 3.0

    def test_session_absent_from_fits_holds_no_hmm(self, tmp_path):
        ps = self._session(tmp_path)
        import_ddm_hmm.import_session(ps, {3: _fit(['eid-b'])})
        assert self._stored_ks(ps.filepath) == []

    def test_mismatch_raises_and_writes_nothing(self, tmp_path):
        ps = self._session(tmp_path)
        fit = _fit([self.EID])
        fit['trials'].loc[0, 'choice'] = -1
        with pytest.raises(ValueError, match='choice'):
            import_ddm_hmm.import_session(ps, {3: fit})
        assert self._stored_ks(ps.filepath) == ['ddm-k9']

    def test_no_stored_trials_raises(self, tmp_path):
        ps = self._session(tmp_path, trials=False)
        with pytest.raises(ValueError, match='trials'):
            import_ddm_hmm.import_session(ps, {3: _fit([self.EID])})
        assert self._stored_ks(ps.filepath) == ['ddm-k9']


def test_parse_args_subjects():
    args = import_ddm_hmm.parse_args(['--subject', 'ZFM-04019'])
    assert args.subject == ['ZFM-04019']
    assert import_ddm_hmm.parse_args([]).subject is None
