# Dataset provenance and license

FinRAG Auditor uses **FinQA** (Chen et al., EMNLP 2021), downloaded from the
authors' official repository: <https://github.com/czyssrs/FinQA>.

- Paper: <https://aclanthology.org/2021.emnlp-main.300/>
- Project site: <https://finqasite.github.io/>
- Dataset source: `czyssrs/FinQA`, `dataset/dev.json` and `dataset/test.json`
- Dataset license: **Creative Commons Attribution 4.0 International (CC BY 4.0)**,
  as stated by the official project site.
- Repository code license: **MIT**, as stated in the official repository's root
  `LICENSE`. This is distinct from the dataset-specific statement on the project site.

The raw JSON files are downloaded locally and ignored by Git. `scripts/download_finqa.py`
pins their SHA-256 hashes so the evaluated split is traceable.

When redistributing derived dataset content, preserve FinQA attribution and comply
with CC BY 4.0. The FinRAG Auditor source code has its own MIT license.

