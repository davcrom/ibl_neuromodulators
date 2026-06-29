# %%
from one.api import ONE

one = ONE()

# %% find the photometry sessions and show subjects
django = [
    "users__username,laura.silva",
    "lab__name,mainenlab",
    "projects__name__icontains,ibl_fibrephotometry",
    "data_dataset_session_related__name__icontains,lightning",
]
sessions = one.alyx.rest("sessions", "list", django=django)

# %%
# genotype and session count per subject
for subject in sorted({session["subject"] for session in sessions}):
    eids = [session["id"] for session in sessions if session["subject"] == subject]
    genotype = one.alyx.rest("subjects", "read", subject)["line"]
    print(subject, genotype, len(eids))

# %% pick a subject
subject = "ZFM-09365" # 5HT
subject = "ZFM-09343" # DA

genotype = one.alyx.rest('subjects','read', subject)['line']
eids = [session["id"] for session in sessions if session["subject"] == subject]

for eid in eids:
    print(one.eid2ref(eid)['date'], eid)

# %% verify that lightningpose can be loaded
from data_loaders import load_pose
from tqdm import tqdm
res = {}
for eid in tqdm(eids):
    try:
        load_pose(one=one, eid=eid)
        res[eid] = True
    except:
        res[eid] = False

# %%
for eid in eids:
    if res[eid]:
        print(eid)

# %% verify brain regions
from data_loaders import get_brain_regions
brain_regions = {}
for eid in tqdm(eids):
    brain_regions[eid] = get_brain_regions(eid=eid, one=one)
    
# %% save valid eids to file
from pathlib import Path

valid_eids = [eid for eid in eids if res[eid]]
session_file = Path(__file__).parent / f"{subject}-sessions.txt"
session_file.write_text("\n".join(valid_eids) + "\n")
print(f"Saved {len(valid_eids)}/{len(eids)} sessions to {session_file}")
# %%
