from dataset_authorization import verify_hf_auth_and_access
from datasets import load_dataset
import glob
import json
import os

# Verify Authorization
verify_hf_auth_and_access()

# Load Hugging Face dataset
ds = load_dataset("lmsys/lmsys-chat-1m")

# Make directory for dataset
output_dir = "./local_datasets/lmsys_chat_1m"
os.makedirs(output_dir, exist_ok=True)

# Save dataset to disk
ds.save_to_disk(output_dir)

print(f"Successfully saved dataset to directory {output_dir}")

# Rename files (train_1.arrow, train_2.arrow, ...).
# The shard count depends on the datasets version, so discover it instead of assuming 6,
# and update state.json so datasets.load_from_disk() still finds the renamed shards.
train_dir = os.path.join(output_dir, "train")
shards = sorted(glob.glob(os.path.join(train_dir, "data-*-of-*.arrow")))
renamed = {}
for i, old_path in enumerate(shards):
    new_name = f"train_{i + 1}.arrow"
    os.rename(old_path, os.path.join(train_dir, new_name))
    renamed[os.path.basename(old_path)] = new_name

state_path = os.path.join(train_dir, "state.json")
if renamed and os.path.exists(state_path):
    with open(state_path) as f:
        state = json.load(f)
    for entry in state.get("_data_files", []):
        entry["filename"] = renamed.get(entry["filename"], entry["filename"])
    with open(state_path, "w") as f:
        json.dump(state, f, indent=2)

print(f"Renamed {len(renamed)} shard(s) in {train_dir}")
