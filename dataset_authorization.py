from huggingface_hub import HfApi, login, get_token
from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError, HfHubHTTPError
from requests.exceptions import HTTPError
import sys

DATASET_ID = "lmsys/lmsys-chat-1m"
DATASET_URL = "https://huggingface.co/datasets/lmsys/lmsys-chat-1m"

def verify_hf_auth_and_access(repo_id: str = DATASET_ID) -> None:
    # Check for stored token or prompt interactive login
    if get_token() is None:
        print("\n[!] Hugging Face access token not found.")
        print("Running interactive login...\n")
        try:
            # Triggers Python equivalent of Terminal's `hf auth login`
            login()
        except Exception as err:
            print(f"\n[X] Login failed: {err}")
            print("Run 'hf auth login' in your terminal and try again.")
            sys.exit(1)

    # Verify account authorization against the gated dataset
    api = HfApi()
    try:
        api.dataset_info(repo_id=repo_id)
        print(f"Success: Account is authorized to use '{repo_id}'.")
    except (GatedRepoError, RepositoryNotFoundError):
        _show_authorization_warning()
        sys.exit(1)
    except (HTTPError, HfHubHTTPError) as err:
        if hasattr(err, "response") and err.response is not None and err.response.status_code in (401, 403):
            _show_authorization_warning()
            sys.exit(1)
        raise err

def _show_authorization_warning() -> None:
    print("\n" + "=" * 78)
    print(f"Access Denied: Your account is not authorized to access this dataset.")
    print(f"Please visit the dataset page in your browser to request access:")
    print(f"  --> {DATASET_URL}")
    print("Once access is granted on Hugging Face, re-run this script.")
    print("=" * 78 + "\n")

if __name__ == "__main__":
    verify_hf_auth_and_access()