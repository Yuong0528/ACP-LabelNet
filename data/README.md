# Data dictionary

All files are comma-separated UTF-8 tables. Sequence strings use the 20 standard one-letter amino-acid codes.

## Stage 1

| File | Rows | Columns | Purpose |
|---|---:|---|---|
| `stage1/set1_train.csv` | 1,273 | `sequence`, `label` | ACP versus antimicrobial peptide; model development |
| `stage1/set1_test.csv` | 319 | `sequence`, `label` | Set 1 independent test |
| `stage1/set2_train.csv` | 1,400 | `sequence`, `label` | ACP versus Swiss-Prot peptide; model development |
| `stage1/set2_test.csv` | 349 | `sequence`, `label` | Set 2 independent test |

`label=1` denotes an anticancer peptide and `label=0` denotes the negative class.

## Stage 2

| File | Rows | Purpose |
|---|---:|---|
| `stage2/train.csv` | 659 | Five-fold development set |
| `stage2/test.csv` | 164 | Fixed independent test set |

Required columns:

- `sequence`: peptide sequence;
- `breast`, `lung`, `colon`, `cervical`, `skin`: binary activity annotations in the model output order;
- `source`: originating database after record harmonization (`cancerppd2`, `dctpep`, or `both`);
- `length`: peptide length;
- `n_labels`: number of positive cancer-type labels.

The complete Stage 2 benchmark contains 823 unique peptide sequences. The split contains no sequence overlap. Positive-label counts over the complete benchmark are 491 breast, 409 lung, 282 colon, 70 cervical, and 206 skin.

The independent test table must not be used for graph construction, augmentation, model selection, hyperparameter selection, or decision-threshold selection.

