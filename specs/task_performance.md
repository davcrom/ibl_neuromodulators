# Spec: Task Performance Analysis

## Objective

Analyze behavioral performance across sessions to characterize:
1. Learning progression (sessions to reach each training stage)
2. Session-level performance metrics
3. Psychometric function parameters

## Questions

### Sessions to Training Stage

1. How should "sessions to reach biased" be computed?
   - Option A: Count of training sessions before first biased session
   - Option B: Total sessions (any type) before first biased session
   - Option C: Something else?

2. Should we only count consecutive sessions, or all sessions regardless of gaps?

3. What about subjects who never reached biased/ephys - exclude or mark as NaN?

### Performance Metrics

4. What performance metrics are needed?
   - Overall fraction correct?
   - Performance on easy trials (≥50% contrast)?
   - Performance by contrast level?
   - Performance by block type (50/20/80 probability)?

5. Should performance be computed on all trials or exclude certain trial types (e.g., no-go, timeouts)?

### Psychometric Fits

6. Should psychometric functions be fit:
   - Option A: Overall (all trials pooled)
   - Option B: Per block type (50%, 20%, 80% probability left)
   - Option C: Both

7. Which parameters to extract?
   - Bias (shift)
   - Threshold (slope)
   - Lapse rates (high/low)
   - Goodness of fit?

### Output

8. Where should results be stored?
   - Add columns to sessions_clean.pqt?
   - Separate file?

9. What visualizations are needed?
   - Learning curves (sessions to stage)?
   - Psychometric curves per session?
   - Performance over sessions?

## Current Implementation

The current script (`scripts/task_performance.py`) implements:
- Sessions to stage: counts training sessions before first biased (Option A above)
- Performance: overall fraction correct + easy trials
- Psychometric: overall fit with 4 parameters (bias, threshold, lapse_high, lapse_low)
- Output: adds columns to sessions_clean.pqt

## Proposed Design

*To be filled after clarification*

