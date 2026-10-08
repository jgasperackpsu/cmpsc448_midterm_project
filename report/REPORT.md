# Fingerprinting LLM Families from Their Text with CNNs and RNNs

**DS340W — Midterm Project Report · Fall 2026**
**Team:** _[Name 1], [Name 2], [Name 3]_

> **Draft status.** Sections 1 and 2 are complete. Every number in Sections 3 and 4 marked
> `[TBD]` comes from `results/` after running `python research_questions.py --rq 1 2 3 4`;
> the matching file is named next to each one. Section 5 is a starter — rewrite it in your own words.

---

## 1. Problem definition and dataset curation (20 pts)

### 1.1 Problem

Text written by large language models is everywhere, but it rarely says which model
produced it. Knowing the source matters for auditing (which vendor's model wrote this
content?), for detecting policy violations by a specific deployment, and for understanding
how much "personality" each model family carries. We frame this as **closed-set,
multi-class text classification**: given a piece of text, predict which of *K* = 6 LLM
families produced it.

Two complications shape the study:

1. **Self-identification is a shortcut.** Many responses contain phrases such as *"As an
   AI language model developed by OpenAI…"*. A classifier that keys on those strings is not
   learning anything interesting, so we remove them before training.
2. **Prompts may leak identity.** On a chatbot platform, people do not send the same prompts
   to every model, so the prompt alone might already reveal the model. Conversely, the prompt
   might add useful context to the response.

### 1.2 Research questions

| | Question |
|---|---|
| **RQ1** | With self-identification removed, how accurately can a CNN and a BiLSTM identify the LLM family from the **response alone**? |
| **RQ2** | Does the **prompt** leak model identity on its own, and does concatenating prompt + response improve on response-only classification? |
| **RQ3** *(extra)* | Do the classifiers learn model-wide fingerprints, or domain-specific patterns that fail to transfer across task categories? |
| **RQ4** *(extra)* | Is the signal mainly **structural** (markdown formatting) or **linguistic** (word choice)? |

### 1.3 Source data

We use **LMSYS-Chat-1M** (Zheng et al., 2023): one million real conversations collected from
the Vicuna demo and Chatbot Arena between April and August 2023, covering 25 models. Each
record has a conversation ID, the model name, the full conversation as a list of
`{role, content}` turns, a detected language, OpenAI moderation output per turn, and a flag
for PII redaction. Access is gated; `dataset_authorization.py` verifies the user's Hugging
Face credentials before `load_data.py` downloads the data.

### 1.4 Family definition

We group model versions into six families so each class reflects a vendor's style rather
than a single checkpoint:

| Family | Model versions in LMSYS-Chat-1M |
|---|---|
| GPT | gpt-3.5-turbo, gpt-4 |
| Claude | claude-1, claude-2, claude-instant-1 |
| LLaMA-2 | llama-2-7b-chat, llama-2-13b-chat |
| Vicuna | vicuna-7b, vicuna-13b, vicuna-33b |
| PaLM | palm-2 |
| MPT | mpt-7b-chat, mpt-30b-chat |

### 1.5 Curation pipeline (`data_cleaning.py`)

| Step | Rule | Why |
|---|---|---|
| Language | keep `language == "English"` | single tokenizer and vocabulary |
| Turn | first user prompt + first assistant reply | later turns depend on conversation history |
| Moderation | drop if either message was flagged | removes abusive / jailbreak content that is unrepresentative of normal behaviour |
| Length | response ≥ 5 words | very short replies ("Sure!") carry almost no signal |
| De-duplication | one response per (family, normalised prompt) | LMSYS users re-send the same test prompts; duplicates would leak across splits |
| Sampling | 4,000 random pairs per family (seed 42) | balanced classes, so accuracy and macro-F1 are directly interpretable (chance = 1/6) |
| Self-ID stripping | drop any sentence that names a model/vendor **and** refers to the speaker | removes the identity shortcut |
| Masking | replace remaining model/vendor names in prompts and responses with `[MODEL]` | removes residual name leakage (e.g. "Hi ChatGPT, …") |
| Task category | heuristic keyword rules → coding / math / creative / general | needed for RQ3 |

Each observation is stored as `(LLM_name, LLM_input, LLM_output)` plus the conversation ID,
model version, task category and the number of self-identification sentences removed
(`data/cleaned/lmsys_clean.csv`).

We deliberately **kept** generic disclaimers that name no model (*"As an AI language model,
I cannot…"*). They are a stylistic habit, not a statement of identity, and removing them
would blur the line between de-identification and style removal.

**Dataset statistics** _(from the `data_cleaning.py` console output and `results/eda/`)_

| | GPT | Claude | LLaMA-2 | Vicuna | PaLM | MPT |
|---|---|---|---|---|---|---|
| Pairs | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| % responses with self-ID removed | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| Median response length (tokens) | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |

![Observations per family and split](../results/eda/split_counts.png)
![Response length by family](../results/eda/response_length_by_family.png)
![Task category by family](../results/eda/task_category_by_family.png)

### 1.6 Pre-processing (`data_processing.py`)

1. **Split first.** A stratified 70 / 15 / 15 train / validation / test split by family
   (seed 42) is made *before* any vocabulary statistics are computed.
2. **Tokenize.** Lower-cased NLTK `word_tokenize` (spaCy's rule-based tokenizer is available
   with `--tokenizer spacy`). NLTK keeps markdown symbols such as `#`, `*` and backticks as
   tokens, which matters for RQ4.
3. **Vocabulary.** Built from training prompts and responses only; tokens seen at least twice,
   capped at 30,000. Index 0 = `<pad>`, 1 = `<unk>`, 2 = `<sep>`. One vocabulary is shared by
   all sequence configurations so that RQ2 compares like with like.
   Vocabulary size: **[TBD]** · validation OOV rate: **[TBD]** (`data/processed/processing_config.json`).
4. **Pad / truncate.** Prompts to 128 tokens, responses to 256, prompt+response to
   128 + 1 + 256 = 385 (`[prompt] <sep> [response]`). Sequences keep their beginning and are
   right-padded. The 256-token response limit covers the median response
   (p50 = **[TBD]**, p95 = **[TBD]**; **[TBD]%** of responses are truncated).

---

## 2. CNN / RNN implementation and training (20 pts)

### 2.1 Shared input representation

Both models start from the same `TokenEmbedding` block: a learned `nn.Embedding(V, 128,
padding_idx=0)` initialised randomly. Sharing the configuration means differences between
the two models come from the sequence encoder, not the representation.

### 2.2 TextCNN (`models.TextCNN`)

```
token ids (B, L)
  → Embedding (B, L, 128) → transpose (B, 128, L)
  → parallel Conv1d(128 → 100, kernel k) for k in kernel_sizes, each + ReLU
  → global max-pool over time on each feature map → (B, 100) each
  → concatenate → (B, 100 · |kernel_sizes|)
  → Dropout(0.5) → Linear(→ K)
```

Each kernel size acts as an n-gram detector (k = 2–5 tokens); global max pooling keeps the
strongest match anywhere in the text, so the CNN is position-invariant. With kernel sizes
(3, 4, 5) and V = 30k the model has ≈ **[TBD]** parameters, most of them in the embedding.

### 2.3 Bidirectional LSTM (`models.BiLSTMClassifier`)

```
token ids (B, L), lengths (B)
  → Embedding (B, L, 128)
  → pack_padded_sequence (padding ignored)
  → 1-layer bidirectional LSTM, hidden H ∈ {128, 256}
  → concat final forward state h_n[-2] and final backward state h_n[-1] → (B, 2H)
  → Dropout(0.5) → Linear(→ K)
```

Packing matters: without it, the final "forward" state would be computed after reading
dozens of `<pad>` tokens for short responses.

### 2.4 Training procedure (`model_training.py`)

| Setting | Value |
|---|---|
| Loss | `nn.CrossEntropyLoss()` (classes are balanced, so unweighted) |
| Optimizer | AdamW, weight decay 0.01 |
| Gradient clipping | max norm 1.0 |
| Epochs | up to 10, early stopping after 3 epochs without validation macro-F1 improvement |
| Model selection | weights from the epoch with the best validation macro-F1 |
| Batching | shuffled; each batch is trimmed to its longest sequence |
| Seed | 42 (Python, NumPy, PyTorch) |
| Hardware | _[TBD — e.g. Google Colab T4 GPU]_ |

Every epoch we log training loss and accuracy, plus validation loss, accuracy, macro-F1 and
the full confusion matrix (`results/<rq>/<model>_<mode>/history.json`).

### 2.5 Hyper-parameter search

A lightweight grid is run separately for every (model, input configuration) pair, selecting
by validation macro-F1:

| | CNN | RNN |
|---|---|---|
| Learning rate | 1e-3, 3e-4 | 1e-3, 3e-4 |
| Batch size | 32, 64 | 32, 64 |
| Kernel sizes | (3,4,5), (2,3,4,5) | — |
| Hidden dimension | — | 128, 256 |
| Configurations | 8 | 8 |

Selected configurations _(from `results/rq1/summary.csv` and `results/rq2/summary.csv`, column `hp`)_:

| Model | Sequence | lr | batch | kernels / hidden | best epoch |
|---|---|---|---|---|---|
| CNN | response | [TBD] | [TBD] | [TBD] | [TBD] |
| CNN | prompt | [TBD] | [TBD] | [TBD] | [TBD] |
| CNN | prompt + response | [TBD] | [TBD] | [TBD] | [TBD] |
| RNN | response | [TBD] | [TBD] | [TBD] | [TBD] |
| RNN | prompt | [TBD] | [TBD] | [TBD] | [TBD] |
| RNN | prompt + response | [TBD] | [TBD] | [TBD] | [TBD] |

RQ3 and RQ4 reuse the RQ1 hyper-parameters so that differences come from the data, not
from re-tuning.

### 2.6 Evaluation protocol

All headline numbers are on the held-out **test** split, which is touched once per final
model. We report accuracy and macro-F1 with **95% bootstrap confidence intervals** (1,000
resamples of the test set), per-class F1, and row-normalised confusion matrices. Paired
comparisons on the same test rows use an **exact McNemar test**; prompt-only accuracy is
compared to chance (1/6) with a one-sided binomial test.

---

## 3. Results (20 pts)

### 3.1 RQ1 — Baseline detectability from responses

| Model | Test accuracy (95% CI) | Test macro-F1 (95% CI) |
|---|---|---|
| CNN | [TBD] | [TBD] |
| BiLSTM | [TBD] | [TBD] |
| Chance | 0.167 | 0.167 |

_Source: `results/rq1/summary.csv`; McNemar CNN vs RNN: `results/rq1/cnn_vs_rnn_mcnemar.json` (p = [TBD])._

![RQ1 test metrics](../results/rq1/rq1_test_metrics.png)
![CNN learning curves](../results/rq1/cnn_output/learning_curves.png)
![RNN learning curves](../results/rq1/rnn_output/learning_curves.png)
![CNN confusion matrix](../results/rq1/cnn_output/confusion_matrix.png)
![RNN confusion matrix](../results/rq1/rnn_output/confusion_matrix.png)

**Reading the results.** _[TBD: 3–5 sentences. How far above chance are both models? Which
model wins and is the gap significant? Which families are easiest / hardest
(`results/rq1/per_class_f1.csv`)? Which pairs are confused most often in the confusion
matrix? Expect related families — e.g. Vicuna, which was fine-tuned on ChatGPT conversations,
and GPT — to be confused with each other; say whether that happened.]_

**Learning curves.** _[TBD: when does validation loss bottom out? Is there over-fitting
(training loss keeps falling while validation loss rises)? How many epochs did early stopping
keep?]_

### 3.2 RQ2 — What does the prompt add?

| Model | Prompt only | Response only | Prompt + response |
|---|---|---|---|
| CNN accuracy | [TBD] | [TBD] | [TBD] |
| CNN macro-F1 | [TBD] | [TBD] | [TBD] |
| BiLSTM accuracy | [TBD] | [TBD] | [TBD] |
| BiLSTM macro-F1 | [TBD] | [TBD] | [TBD] |

_Source: `results/rq2/summary.csv`, statistics in `results/rq2/analysis.json`._

![RQ2 mode comparison](../results/rq2/rq2_mode_comparison.png)

**Do prompts leak identity?** _[TBD: prompt-only accuracy vs chance and its binomial p-value.
If clearly above chance, the prompt distribution differs between families — e.g. Arena users
send coding questions to GPT-4 more often, or a family was available only during a certain
period. Look at the prompt-only confusion matrices (`results/rq2/*_input/confusion_matrix.png`)
to see which families are predictable from prompts.]_

**Does the prompt help the response?** _[TBD: difference both − response-only in accuracy and
macro-F1, McNemar p-value. If the gain is small and not significant, the response already
carries almost all the signal; if it is significant, say whether the gain could be explained
by prompt leakage rather than better "understanding" of the response.]_

---

## 4. In-depth analyses and experiments (20 pts)

### 4.1 Error analysis

_[TBD — pick ~10 misclassified test responses (`test_predictions.npz` + `data/processed/meta.csv`
give the row ids) and describe patterns: very short responses, code-only answers, refusals,
non-English fragments that slipped through, responses truncated at 256 tokens.]_

### 4.2 Statistical robustness

Bootstrap intervals are reported for every headline number. _[TBD: comment on how wide they
are (~±1–2 points with ~3,600 test rows), and which differences between models/configurations
are larger than the intervals.]_

### 4.3 RQ3 (extra credit) — Do fingerprints transfer across task domains?

Both models were retrained using only **general** prompts (train and validation), with RQ1's
hyper-parameters, then tested on the general test rows (in-domain) and zero-shot on the
coding, math and creative test rows.

| Model | In-domain (general) | Coding | Math | Creative |
|---|---|---|---|---|
| CNN macro-F1 | [TBD] | [TBD] | [TBD] | [TBD] |
| BiLSTM macro-F1 | [TBD] | [TBD] | [TBD] | [TBD] |

_Source: `results/rq3/summary.csv` (also has accuracy, CIs and the drop vs in-domain)._

![RQ3 domain transfer](../results/rq3/rq3_domain_transfer.png)

_[TBD: a small drop means the models learned family-wide habits (tone, hedging, phrasing); a
large drop on coding — where answers are mostly code — means the fingerprint lives in prose.
Note that categories come from keyword heuristics and test subsets are small, so check the
confidence intervals.]_

### 4.4 RQ4 (extra credit) — Structural vs linguistic signal

**Stylometric profile.** For each response we measured length (words), type–token ratio,
moving-average TTR over 50-word windows (MATTR, which is not biased by length), and counts of
markdown headers, bullet/numbered list items, bold spans and fenced code blocks.

![Stylometric features by family](../results/rq4/rq4_style_features.png)

_[TBD: describe the clearest differences, e.g. which family writes the longest answers, which
uses the most markdown, which has the highest lexical diversity. Kruskal–Wallis tests in
`results/rq4/kruskal_wallis.json` show whether each feature differs across families.]_

**How far do eight hand-crafted features get?** A logistic regression on these features alone
reaches accuracy **[TBD]** and macro-F1 **[TBD]** (`results/rq4/feature_only_logreg.json`) —
a useful lower bound for what pure "shape" information gives, compared with [TBD] for the
neural models.

**Structure ablation.** We removed all markdown structure (headers, bullets, numbered-list
markers, bold, code fences, inline backticks, block quotes, table pipes) from every response,
re-tokenized with the same split, and retrained both models with RQ1's hyper-parameters.

| Model | Original accuracy | Stripped accuracy | Drop |
|---|---|---|---|
| CNN | [TBD] | [TBD] | [TBD] |
| BiLSTM | [TBD] | [TBD] | [TBD] |

_Source: `results/rq4/summary.csv`._

![Structure ablation](../results/rq4/rq4_structure_ablation.png)

_[TBD: a small drop means the signal is mostly linguistic; a large drop means formatting
habits carry a lot of it. Also note what stripping cannot remove: response length, line
structure that becomes sentence breaks, and code content itself.]_

### 4.5 Limitations

- **Time-bound data.** LMSYS-Chat-1M reflects mid-2023 model versions; today's GPT, Claude or
  Llama models write differently, so the classifiers would need retraining.
- **Heuristic de-identification.** Sentence-level stripping can miss paraphrased
  self-references and can over-remove sentences that mention a vendor in passing.
- **Heuristic task categories.** Keyword rules mislabel some prompts, which blurs RQ3's domains.
- **Closed set.** The classifiers can only choose among six families; text from any other
  model is forced into one of them.
- **Truncation.** Only the first 256 response tokens are seen; long-form structure is lost.
- **Sampling bias.** Arena users choose which models to talk to and what to ask, which is the
  prompt leakage that RQ2 measures.

---

## 5. Lessons and experience (20 pts)

> **Starter text — rewrite in your own words.** This section is graded on *your* experience.
> Keep what is true for your team, delete what isn't, and add specifics (what broke, how long
> runs took, how you split the work).

- **Shortcut features show up first.** Before stripping self-identification, a classifier
  could score well just by spotting vendor names. Deciding what counts as "identity" versus
  "style" (e.g. keeping *"As an AI language model"*) was a design decision we had to justify,
  not a technicality.
- **Split before you build anything from the data.** Building the vocabulary only from the
  training split, and de-duplicating repeated prompts before splitting, were necessary to keep
  test numbers honest.
- **Padding has consequences.** Packing sequences for the LSTM and trimming batches to their
  longest sequence fixed both correctness (the final hidden state) and speed.
- **Cheap baselines put deep models in context.** The prompt-only models and the
  hand-crafted-feature logistic regression told us more about *where* the signal is than the
  headline accuracy did.
- **Engineering for long runs.** Caching each finished experiment so an interrupted grid search
  could resume saved _[TBD]_ hours of recomputation.
- _[Add: how the team divided work, what you would do differently, what you would try next —
  e.g. a pretrained transformer encoder, newer models, or open-set detection of unseen families.]_

---

## References

- Zheng, L., Chiang, W.-L., Sheng, Y., et al. (2023). *LMSYS-Chat-1M: A Large-Scale Real-World
  LLM Conversation Dataset.* arXiv:2309.11998.
- Kim, Y. (2014). *Convolutional Neural Networks for Sentence Classification.* EMNLP.
- Hochreiter, S., & Schmidhuber, J. (1997). *Long Short-Term Memory.* Neural Computation, 9(8).
- Loshchilov, I., & Hutter, F. (2019). *Decoupled Weight Decay Regularization.* ICLR.
- Covington, M. A., & McFall, J. D. (2010). *Cutting the Gordian Knot: The Moving-Average
  Type–Token Ratio (MATTR).* Journal of Quantitative Linguistics, 17(2).
