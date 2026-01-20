# Spec: Photometry QC Pipeline Refactoring

## Objective

Refactor the photometry QC pipeline to run quality control efficiently and robustly over all sessions, with proper error logging and data validation.

## Background

### Current State

**PhotometrySession.run_qc()** (`iblnm/data.py:129-196`):
- Calls `iblphotometry.qc.qc_series()` for raw and sliding metrics
- No error handling - exceptions propagate to caller
- Assumes data is already loaded successfully

**photometry_qc.py** (`scripts/photometry_qc.py`):
- Loads sessions from hardcoded file path
- Hardcoded filter to single NM (ACh)
- Uses `iblphotometry.qc.run_qc()` directly with parallel processing
- No session cleaning (excluded subjects/types not filtered)
- No error logging
- Contains ~200 lines of old/archived code mixed with working code (lines 225-452)

**Existing error handling** (`iblnm/io.py`):
- `@exception_logger` decorator captures errors to list instead of raising
- Used by: `unpack_session_dict`, `get_extended_qc`, `get_subject_info`, `get_datasets`, `get_target_regions`
- Pattern: functions take optional `exlog` parameter

### Problems

1. No validation that data can actually be retrieved before running QC
2. `data_complete` flag checks metadata presence, not actual retrievability
3. No error log collected when sessions fail
4. Excluded subjects/session types not filtered before QC
5. Script is messy - hard to understand what's current vs archived

## Requirements

### Functional

1. **Session filtering**: Apply `clean_sessions()` before QC to exclude known bad subjects/session types

2. **Validate-as-you-go**: During the QC loop, for each session:
   - Attempt to load photometry signal and locations
   - If loading fails: log error, update session flags, skip to next session
   - If loading succeeds: run QC

3. **New columns** in df_sessions (updated during QC, cached in sessions.pqt):
   - `can_load_trials` (bool): trials table actually loads
   - `can_load_photometry_signal` (bool): photometry signal actually loads
   - `can_load_photometry_locations` (bool): photometry locations actually loads
   - `load_error` (str | None): error details if loading fails

   Note: Use `can_load_*` prefix to distinguish from metadata-based flags like `has_extracted_task`

4. **Error logging**: Collect all errors (loading + QC) in structured list:
   ```python
   {'eid': str, 'subject': str, 'error_type': str, 'error_message': str, 'stage': str}
   ```
   Where `stage` is 'load_signal', 'load_locations', 'load_trials', or 'qc'

5. **Archived code**: Move old code (lines 225-452) to `snippets/photometry_qc_archive.py`

### Non-functional

1. **Efficiency**: Single pass - validate and QC in same loop
2. **Robustness**: Single session failure logs error and continues
3. **Clarity**: Clean script with clear workflow

## Design

### QC loop structure (in photometry_qc.py)

```python
error_log = []

for idx, session in tqdm(df_sessions.iterrows()):
    eid = session['eid']

    # Initialize flags
    df_sessions.loc[idx, 'can_load_photometry_signal'] = False
    df_sessions.loc[idx, 'can_load_photometry_locations'] = False
    df_sessions.loc[idx, 'can_load_trials'] = False
    df_sessions.loc[idx, 'load_error'] = None

    # Try to load photometry signal
    try:
        signal = one.load_dataset(eid, 'photometry.signal.pqt', collection='alf/photometry')
        df_sessions.loc[idx, 'can_load_photometry_signal'] = True
    except Exception as e:
        error_log.append({'eid': eid, 'subject': session['subject'],
                        'error_type': type(e).__name__, 'error_message': str(e),
                        'stage': 'load_signal'})
        df_sessions.loc[idx, 'load_error'] = f"signal: {type(e).__name__}"
        continue

    # Try to load photometry locations
    try:
        locations = one.load_dataset(eid, 'photometryROI.locations.pqt', collection='alf/photometry')
        df_sessions.loc[idx, 'can_load_photometry_locations'] = True
    except Exception as e:
        error_log.append({'eid': eid, 'subject': session['subject'],
                        'error_type': type(e).__name__, 'error_message': str(e),
                        'stage': 'load_locations'})
        df_sessions.loc[idx, 'load_error'] = f"locations: {type(e).__name__}"
        continue

    # Try to load trials (optional, for tracking)
    try:
        trials = one.load_dataset(eid, '_ibl_trials.table.pqt', collection='alf/task_00')
        df_sessions.loc[idx, 'can_load_trials'] = True
    except:
        try:
            trials = one.load_dataset(eid, '_ibl_trials.table.pqt', collection='alf')
            df_sessions.loc[idx, 'can_load_trials'] = True
        except:
            pass  # trials not required for QC

    # Run QC
    try:
        qc_result = run_qc_for_session(eid, signal, ...)
        qc_results.append(qc_result)
    except Exception as e:
        error_log.append({'eid': eid, 'subject': session['subject'],
                        'error_type': type(e).__name__, 'error_message': str(e),
                        'stage': 'qc'})
```

### Updated script workflow

1. Load sessions from `SESSIONS_FPATH`
2. Apply `clean_sessions()`
3. Run QC loop (validate + QC per session, collect errors)
4. Save updated df_sessions with `can_load_*` columns
5. Save QC results
6. Save error log

### New file: `snippets/photometry_qc_archive.py`

Move archived code from photometry_qc.py (eval_metric, qc_series, response similarity matrix).

## Files to Modify

- `scripts/photometry_qc.py`: Refactor to validate-as-you-go workflow
- `snippets/photometry_qc_archive.py`: New file with archived code

## Verification

1. Run on small subset first (e.g., 10 sessions)
2. Check error_log captures failures with useful info
3. Check df_sessions `can_load_*` columns are populated correctly
4. Run on full dataset, verify no crashes
5. Compare `can_load_*` results with metadata-based `has_*` flags
