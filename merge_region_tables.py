import numpy as np
import pandas as pd

# Region mappings parsed from filenames in Feb2025 folder
df1 = pd.read_csv('metadata/regions_fromfilenames.csv')
# Region mappings from 'Mice performance tables 100' google sheet
df2 = pd.read_csv('metadata/regions_mantable.csv')

# Outer merge with indicator to preserve all rows from both dataframes.
df_merged = pd.merge(
    df1, df2, 
    on=['subject', 'date', 'eid'], 
    how='outer', 
    suffixes=('_x', '_y'),
    indicator=True
)

# Find rows that appear in both files but have mismatching ROI labels
df_tmp = df_merged.dropna()
df_mismatch = df_tmp[df_tmp['ROI_x'] != df_tmp['ROI_y']]
# Check that all mismatching labels come from sessions with multiple fibers
df_insertions = pd.read_csv('metadata/website.csv')  # table provided by KB/GC/OW
def _get_N_rois(series, df_insertions):
    eid = series['eid']
    sess = df_insertions.query('eid == @eid')
    if len(sess) != 1:
        # Note: ~10 sessions have no entries in this file, none have duplicate entries
        ## TODO: check if these are all from the same animal
        return
    return sess['N rois'].values[0]
n_rois = df_mismatch.apply(_get_N_rois, df_insertions=df_insertions, axis='columns')
assert all(n_rois.dropna() > 1)

# Select rows that were present in both tables
df_both = df_merged[df_merged['_merge'] == 'both']
# Melt the two ROI columns into one, then drop duplicate rows
df_melted = df_both.melt(
    id_vars=['subject', 'date', 'eid'],
    value_vars=['ROI_x', 'ROI_y'],
    var_name='ROI_side',
    value_name='ROI'
).drop_duplicates(subset=['subject', 'date', 'eid', 'ROI']).drop(columns='ROI_side')

# Select rows that were unique to one of the two tables
df_only = df_merged[df_merged['_merge'] != 'both'].copy()
# Take the non-null value
df_only['ROI'] = df_only['ROI_x'].combine_first(df_only['ROI_y'])
# Keep only the necessary columns
df_only = df_only[['subject', 'date', 'eid', 'ROI']]

# Concatenate the two results
df_final = pd.concat([df_melted, df_only], ignore_index=True)

# Confirm that each session has no duplicate region labels 
for idx, group in df_final.groupby('eid'):
    if len(group) == 1:
        continue
    else:
        assert len(group['ROI'].unique()) == len(group)

# Save the result
df_final.to_csv('metadata/regions.csv')