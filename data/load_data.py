from dataset_authorization import verify_hf_auth_and_access, _show_authorization_warning
from datasets import load_dataset
import os
import json
import shutil

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

# Rename files
for i in range(6):
    old_name = f"train/data-{i:05d}-of-00006.arrow"
    new_name = f"train/train_{i+1}.arrow"

    old_path = os.path.join(output_dir, old_name)
    new_path = os.path.join(output_dir, new_name)
    
    if os.path.exists(old_path):
        os.rename(old_path, new_path)

train_dir = os.path.join(output_dir, "train")

# Path to state.json in train subfolder
old_state_path = os.path.join(train_dir, "state.json")
new_state_path = os.path.join(output_dir, "state.json")
old_info_path = os.path.join(train_dir, "dataset_info.json")
new_info_path = os.path.join(output_dir, "dataset_info.json")

# Move state.json to the root folder
if os.path.exists(old_state_path):
    shutil.move(old_state_path, new_state_path)
    print(f"Moved state.json to {output_dir}")

if os.path.exists(old_info_path):
    shutil.move(old_info_path, new_info_path)
    print(f"Moved dataset_info.json to {output_dir}")

# Updating state.json
state_json_path = os.path.join(output_dir, "state.json")
with open(state_json_path, "r") as f:
    state_data = json.load(f)

state_data["_data_files"] = [
    {"filename": f"train_{i+1}.arrow"} for i in range(6)
]

with open(state_json_path, "w") as f:
    json.dump(state_data, f, indent=2)