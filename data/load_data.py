from dataset_authorization import verify_hf_auth_and_access, _show_authorization_warning
from datasets import load_dataset
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