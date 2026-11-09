# LHJ rebuild

Reproducible code package for the experiment **“Bridging Brains and Large Language Models: Multimodal ERP Augmentation for Novelty Classification of Metaphors.”**

This repository contains the refactored training implementation and its configuration templates. It does not contain the private EEG/ERP dataset, pretrained-model weights, experiment outputs, credentials, LaTeX sources, figures, or PDFs.

The package is designed for researchers who receive the code and need to reproduce the experiment on their own workstation or remote server. Before running anything, replace the three virtual path placeholders described below with paths available in your environment.

## 1. Scientific overview

The task is three-class Chinese metaphor-novelty classification. Each example combines:

1. a Chinese metaphor sentence;
2. a novelty label encoded as `Score` values `1`, `2`, or `3`;
3. four pre-extracted ERP features:
   - `N400_peak`;
   - `N400_trough`;
   - `LPC_peak`;
   - `LPC_trough`.

The model uses a local Hugging Face-compatible text encoder and a late-fusion ERP branch. Text representations and ERP features are projected into a shared hidden dimension and concatenated before a three-class classifier. LoRA adaptation is enabled by default and can be disabled with `--no-lora`.

The implementation is intended to reproduce the multimodal ablation setting described by the accompanying research materials. It is not a release of the private human EEG dataset.

## 2. Repository layout

```text
LHJ rebuild/
├── configs/
│   └── paths.example.env       # virtual path template; no private paths
├── scripts/
│   ├── check_package.py        # package safety check
│   └── run_train.sh            # environment-variable based launcher
├── src/
│   ├── data.py                 # Excel schema, labels, loaders, ablations
│   ├── model.py                # text encoder + ERP late fusion
│   └── train_multimodal.py     # training, validation, testing, outputs
├── .gitignore
├── requirements.txt
└── README.md
```

## 3. Shared path contract

All scripts use the same three virtual path names. Keep these names unchanged when chaining scripts together:

```text
YOUR_DATA_DIR    input directory containing the Excel splits
YOUR_MODEL_DIR   local pretrained-model directory
YOUR_OUTPUT_DIR  shared root directory for generated results
```

The training script reads from `YOUR_DATA_DIR` and `YOUR_MODEL_DIR`, and writes to `YOUR_OUTPUT_DIR`. If another script consumes the training results, it must use the same `YOUR_OUTPUT_DIR` value rather than defining a second result-root placeholder.

The generated files are organized as follows:

```text
YOUR_OUTPUT_DIR/
└── <ABLATION>/
    ├── best_model.pt
    └── metrics.csv
```

`YOUR_DATA_DIR`, `YOUR_MODEL_DIR`, and `YOUR_OUTPUT_DIR` are deliberately virtual names. Replace their values only in a local, untracked configuration file or in the shell environment. Do not commit private machine paths.

## 4. Expected data layout

The default split-based run expects these files inside `YOUR_DATA_DIR`:

```text
YOUR_DATA_DIR/train_set.xlsx
YOUR_DATA_DIR/val_set.xlsx
YOUR_DATA_DIR/test_set.xlsx
```

Each file must contain the following columns:

```text
Sentence
Score
N400_peak
N400_trough
LPC_peak
LPC_trough
```

Data requirements:

- `Sentence` contains the input text.
- `Score` contains exactly one of `1`, `2`, or `3`.
- The four ERP columns must be numeric or convertible to floating-point values.
- The training, validation, and test split must be defined before training.
- Sentence-level separation should be preserved to avoid leakage between splits.

The loader converts `Score` to zero-based labels (`1 → 0`, `2 → 1`, `3 → 2`). It validates the required schema before training.

## 5. Environment setup

The implementation requires Python 3.10 or newer. Create an isolated environment on the target machine and install the minimum dependencies listed in `requirements.txt`:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The pretrained model must already be available locally because the implementation uses `local_files_only=True`. The model directory must be compatible with `transformers.AutoTokenizer`, `transformers.AutoConfig`, and `transformers.AutoModel`.

## 6. Configure the paths

Create a local configuration file that is ignored by Git:

```bash
cp configs/paths.example.env configs/paths.env
```

Edit `configs/paths.env` so that it contains your own paths:

```bash
DATA_DIR=YOUR_DATA_DIR
MODEL_DIR=YOUR_MODEL_DIR
OUTPUT_DIR=YOUR_OUTPUT_DIR
```

The values above are placeholders. Replace them with valid paths on the machine where the experiment will run. Do not paste private paths into the public repository.

Load the configuration into the current shell:

```bash
set -a
source configs/paths.env
set +a
```

## 7. Reproduce the default experiment

Run the provided launcher from the repository root:

```bash
python scripts/check_package.py
bash scripts/run_train.sh
```

The launcher uses the shared variables `DATA_DIR`, `MODEL_DIR`, and `OUTPUT_DIR`, which correspond to the virtual placeholders `YOUR_DATA_DIR`, `YOUR_MODEL_DIR`, and `YOUR_OUTPUT_DIR`.

The equivalent explicit command is:

```bash
python -m src.train_multimodal \
  --data-dir "$DATA_DIR" \
  --model-dir "$MODEL_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --ablation full \
  --epochs 15 \
  --batch-size 8 \
  --grad-accum 1 \
  --seed 12
```

The code automatically selects CUDA when available and otherwise runs on CPU. Training is not performed by the package-safety check.

## 8. Ablation settings

The `--ablation` argument accepts the following settings:

```text
full                 all four ERP features
text_only            no ERP feature contribution
wo_n400              remove N400_peak and N400_trough
wo_lpc               remove LPC_peak and LPC_trough
wo_peak              remove both peak features
wo_trough            remove both trough features
n400_peak_only       retain N400_peak only
n400_trough_only     retain N400_trough only
lpc_peak_only        retain LPC_peak only
lpc_trough_only      retain LPC_trough only
all                  run every registered ablation sequentially
```

For example:

```bash
python -m src.train_multimodal \
  --data-dir "$DATA_DIR" \
  --model-dir "$MODEL_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --ablation n400_trough_only
```

Each condition writes to its own subdirectory under the same `OUTPUT_DIR` root. This preserves a consistent input/output contract for downstream analysis.

## 9. Cross-validation files

If a separate cross-validation directory is used, pass the corresponding filenames explicitly while keeping the same `DATA_DIR` placeholder:

```bash
python -m src.train_multimodal \
  --data-dir "$DATA_DIR" \
  --train-file train.xlsx \
  --val-file VAL.xlsx \
  --test-file test.xlsx \
  --model-dir "$MODEL_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --ablation full
```

The package does not assume that cross-validation files have the same names as the default split files. Confirm the filenames and schema before running.

## 10. Outputs and evaluation

For each ablation, the training script writes:

```text
<OUTPUT_DIR>/<ablation>/best_model.pt
<OUTPUT_DIR>/<ablation>/metrics.csv
```

The best checkpoint is selected by validation accuracy. The metrics file records training, validation, and optional test metrics for each epoch:

- loss;
- accuracy;
- macro precision;
- macro recall;
- macro F1.

To use a generated result in a later analysis script, set that script's result-root input to the same `YOUR_OUTPUT_DIR` value and select the relevant ablation subdirectory.

## 11. Reproducibility notes

The default seed is `12`. The code seeds Python, NumPy, and PyTorch and enables deterministic CuDNN behavior when CUDA is available. Exact numerical replication can still vary with GPU hardware, CUDA/cuDNN versions, Transformer implementation details, and the exact local model checkpoint.

For a reproducible report, record:

1. the Git commit used for the run;
2. Python and package versions;
3. the local model identifier and checksum;
4. GPU and CUDA information;
5. the exact data split and column schema;
6. the command-line arguments;
7. the output directory and ablation name.

The private dataset and model files should be versioned and backed up according to the applicable ethics, privacy, and data-management requirements. They should not be committed to this repository.

## 12. PyCharm and remote execution

The code is compatible with a PyCharm remote-interpreter workflow. Configure the following items in PyCharm with machine-specific values:

- Python interpreter;
- deployment directory;
- `YOUR_DATA_DIR`;
- `YOUR_MODEL_DIR`;
- `YOUR_OUTPUT_DIR`.

The source code does not require editing when moved between machines. Only the local environment configuration and command-line path values need to change.

## 13. Package verification

Before committing or uploading, run:

```bash
python scripts/check_package.py
```

The check verifies required source files and rejects common dataset and model-weight extensions, including `.xlsx`, `.csv`, `.pt`, `.pth`, `.bin`, and `.safetensors`.

Also inspect the Git diff and confirm that no credentials, private paths, model files, result files, or human-subject data are present.

## 14. Scope and limitations

This repository provides a reproducible implementation skeleton for the multimodal ERP-text experiment. It does not provide the private dataset, raw EEG recordings, pretrained model weights, historical result tables, or the manuscript source. Reproduction therefore requires authorized access to the same type of data splits and a compatible local pretrained model.

The refactored entry point is the supported execution path. Historical scripts, private deployment files, and machine-specific research artifacts are intentionally excluded from the upload package.
