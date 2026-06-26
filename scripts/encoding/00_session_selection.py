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
    line = one.alyx.rest("subjects", "read", subject)["line"]
    print(subject, line, len(eids))

# %% pick a subject
subject = "ZFM-09343"
genotype = one.alyx.rest('subjects','read', subject)['line']

brain_region = "SNc-l"  # TODO dataset-specific
eids = [session["id"] for session in sessions if session["subject"] == subject]

# %%
# eid = "6931684c-a721-4db8-9698-e3101d0e4a1b" # first session
# # eid = "5e57fcd0-8743-41c8-8360-d846a4e0469d" # last session
# brain_region = "SNc-l"  # TODO dataset-specific

# %%
subject = "ZFM-09365"
genotype = one.alyx.rest('subjects','read', subject)['line']

eids = [session["id"] for session in sessions if session["subject"] == subject]

for eid in eids:
    print(one.eid2ref(eid)['date'], eid)