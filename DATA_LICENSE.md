# Data attribution and terms

The customer-support examples in `data/splits/bitext-v1/*.jsonl` originate from
Bitext Innovations' **Bitext Customer Support LLM Chatbot Training Dataset**.

- Source: https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset
- Pinned revision: `fb86d1b60038970d79fefbfa0c28b47c22d8b961`
- Upstream declared data license: Community Data License Agreement – Sharing, Version 1.0 (CDLA-Sharing-1.0).
- License terms: https://cdla.dev/sharing-1-0/
- Attribution: Bitext Innovations, 2024.

This project redistributes these examples under CDLA-Sharing-1.0. It retains the
source fields, partitions records into train/dev/sealed_test by normalized
instruction groups, and adds source-row, record-ID and template-ID fields.
Details and source checksums are in `docs/data-card.md` and the split manifest.
The tokenizer in `artifacts/tokenizer/bitext-bpe-v1` was trained on the train split;
its provenance is documented in `docs/tokenizer.md`.

The files under `data/derived/` are modified training-target versions of the same
Bitext examples and are also redistributed under CDLA-Sharing-1.0. Their manifests,
rules and change ledgers identify the changes to result/privacy/refund wording.
These edits do not establish real merchant policies or authorized business actions.

This notice covers the dataset; it does not assign a license to the project's
original source code. Public repository visibility alone does not grant a code license.
