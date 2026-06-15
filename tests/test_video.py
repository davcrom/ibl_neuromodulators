"""Tests for scripts/sync_lp_progress.py merge_pose join helper."""
import numpy as np
import pandas as pd

import scripts.sync_lp_progress as sync


def test_merge_pose_left_joins_on_eid():
    df_video = pd.DataFrame({
        'eid': ['eid-0', 'eid-1', 'eid-2'],
        'video_qc_score': [1.0, 0.5, 0.0],
    })
    df_pose = pd.DataFrame({
        'eid': ['eid-0', 'eid-2'],
        'paw_drift': [0.1, 0.3],
        'qc_pose': ['PASS', 'FAIL'],
    })

    result = sync.merge_pose(df_video, df_pose)

    # One row per video eid, no duplication
    assert len(result) == len(df_video)
    assert list(result['eid']) == ['eid-0', 'eid-1', 'eid-2']
    # Pose columns present alongside video columns
    assert {'paw_drift', 'qc_pose', 'video_qc_score'} <= set(result.columns)
    # Matched eids carry pose values
    matched = result.set_index('eid')
    assert matched.loc['eid-0', 'paw_drift'] == 0.1
    assert matched.loc['eid-2', 'qc_pose'] == 'FAIL'
    # Unmatched eid gets NaN pose columns
    assert np.isnan(matched.loc['eid-1', 'paw_drift'])
    assert pd.isna(matched.loc['eid-1', 'qc_pose'])
