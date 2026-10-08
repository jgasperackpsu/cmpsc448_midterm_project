from numpy import isin
import os
import re
import random
from collections import Counter
import pandas as pd
from sklearn.model_selection import train_test_split
import nltk
from nltk.tokenize import word_tokenize
import pyarrow.dataset as ds
import pyarrow.ipc as ipc
import glob
import ssl
import shutil

###############################################
################ DATA LOADING #################
###############################################

# Load in arrow files
arrow_files = sorted(glob.glob("./local_datasets/lmsys_chat_1m/train/*.arrow"))

dataframes = []
for file_path in arrow_files:
    with open(file_path, 'rb') as f:
        # Open as Arrow Stream Reader
        reader = ipc.RecordBatchStreamReader(f)
        table = reader.read_all()
        dataframes.append(table.to_pandas())

lmsys_data = pd.concat(dataframes, ignore_index=True)

print(f"Combined total rows: {len(lmsys_data)}")
print(lmsys_data.head())

# Tokenization assets
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('punkt')
nltk.download('punkt_tab')

old_dir = "/Users/jgasperack/"
new_dir = "./local_datasets/"
old_path = os.path.join(old_dir, "nltk_data")
new_path = os.path.join(new_dir, "nltk_data")

if not os.path.exists(new_path):
    shutil.move(old_path, new_path)

###############################################
################ DATA CLEANING ################
###############################################

model_to_family = {
    # OpenAI Family
    'gpt-4': 'OpenAI',
    'gpt-3.5-turbo': 'OpenAI',
    
    # Anthropic Family
    'claude-1': 'Anthropic',
    'claude-2': 'Anthropic',
    'claude-instant-1': 'Anthropic',
    
    # Meta / Llama Family
    'llama-13b': 'Llama',
    'llama-2-7b-chat': 'Llama',
    'llama-2-13b-chat': 'Llama',
    
    # Vicuna Family
    'vicuna-7b': 'Vicuna',
    'vicuna-13b': 'Vicuna',
    'vicuna-33b': 'Vicuna',
    
    # MPT Family
    'mpt-7b-chat': 'MPT',
    'mpt-30b-chat': 'MPT',
}

filtered_df = lmsys_data[lmsys_data["model"].isin(model_to_family.keys())].copy()
filtered_df["family"] = filtered_df["model"].map(model_to_family)

def extractPair(conversation):
    user_input, assistant_output = None, None
    
    # Check that conversation is iterable (numpy array or list)
    if conversation is not None and len(conversation) > 0:
        for turn in conversation:
            # Safely handle turn access whether it's dict or struct
            if isinstance(turn, dict):
                role = turn.get('role', '')
                content = turn.get('content', '')
            else:
                try:
                    role = turn['role']
                    content = turn['content']
                except Exception:
                    continue

            # Strip whitespace to check for actual non-empty text
            content_str = str(content).strip() if content is not None else ""

            if role == 'user' and not user_input and content_str:
                user_input = content_str
            elif role == 'assistant' and not assistant_output and content_str:
                assistant_output = content_str

            if user_input and assistant_output:
                return user_input, assistant_output
                
    return None, None


print("Extracting response pairs...")
results = [extractPair(conv) for conv in filtered_df["conversation"]]

valid_inds = [i for i, (p,r) in enumerate(results) if p is not None and r is not None]

parsed = pd.DataFrame({
    'llm_name': filtered_df['model'].iloc[valid_inds].values,
    'llm_family': filtered_df['family'].iloc[valid_inds].values,
    'llm_input': [results[i][0] for i in valid_inds],
    'llm_output': [results[i][1] for i in valid_inds]
})

print(f"Extracted valid prompt-response pairs: {len(parsed)}")

SEED = 71

# Self-ID string removal
self_ids = [
    r"I am (Claude|GPT|Chat GPT|Llama|Vicuna|MPT|WizardLM|PaLM|Koala|Alpaca|Dolly|Mpt|ChatGLM|Guanaco|RWKV|StableLM)[^.\n]*[.\n]?",
    r"I am an (AI|Artificial Intelligence) developed by [^,\n]+, \s*",
    r"As a (Large Language Model|LLM) developed by [^,\n]+, \s*",
    r"As an (AI|Artificial Intelligence) (assistant|model|language model), \s*"
    ]

combined_pattern = re.compile('|'.join(self_ids), flags=re.IGNORECASE)

# Random Sampling
SAMPLE_SIZE = 2000
sampled = parsed.groupby("llm_family", group_keys=False).apply(
    lambda x: x.sample(min(len(x), SAMPLE_SIZE), random_state=SEED)
    ).reset_index(drop=True)

def cleanText(text):
    if not isinstance(text, str):
        return ""
    return combined_pattern.sub("", text).strip()

sampled['llm_input'] = sampled['llm_input'].apply(cleanText)
sampled['llm_output'] = sampled['llm_output'].apply(cleanText)

formatted = [
    (row['llm_name'], row['llm_input'], row['llm_output'])
    for i, row in sampled.iterrows()
]

###################################################
################ DATA PREPROCESSING ###############
###################################################

tokens = [
    (name, word_tokenize(i.lower()), word_tokenize(o.lower()))
    for name, i, o in formatted
]

# Build vocab map
MAX_SIZE = 10000
PAD_T = "<PAD>"
UNK_T = "<UNK>"

all_tokens = []
for _, i_t, o_t in tokens:
    all_tokens.extend(i_t)
    all_tokens.extend(o_t)

most_common = [t for t, _ in Counter(all_tokens).most_common(MAX_SIZE - 2)]

vocab = {PAD_T: 0, UNK_T: 1}
for i, t in enumerate(most_common, start=2):
    vocab[t] = i

def tokensToID(tokens, map):
    return [map.get(token, map[UNK_T]) for token in tokens]

MAX_SEQ = 128

def padTrunc(ids, max_l = MAX_SEQ, pad_id=0):
    if len(ids) > max_l:
        return ids[:max_l]
    else:
        return ids + [pad_id] * (max_l - len(ids))

processed_samp = []
for name, i, o in tokens:
    processed_samp.append({
        'llm_name': name,
        'input_ids': padTrunc(tokensToID(i, vocab)),
        'output_ids': padTrunc(tokensToID(o, vocab))
    })

train, temp = train_test_split(
    processed_samp,
    test_size=0.3,
    random_state=SEED
    )

val, test = train_test_split(
    temp,
    test_size=0.5,
    random_state=SEED
    )