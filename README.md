# SR_XAI Local Explanation Optimizer

`SR_XAI` trains and runs a local explanation optimizer for 3D XAI maps. It combines multiple attribution methods, evaluates them with faithfulness and complexity metrics, and produces an optimized explanation map.

Please cite:

- https://arxiv.org/abs/2405.10008
- https://doi.org/10.1016/j.aiopen.2025.02.001

## Project Structure

```text
Local_optimizer/
├── configs/
│   ├── train_config.yaml
│   └── inference_config.yaml
├── functional_testing/
│   ├── toy_xai_pipeline.py
│   ├── toy_results.ipynb
│   └── results/toy_results.json
├── script/
│   ├── train_script.py
│   └── test_script.py
├── SR_XAI/
│   ├── training/
│   │   ├── train.py
│   │   └── runner.py
│   ├── inference/
│   │   ├── test.py
│   │   └── runner.py
│   ├── models/
│   │   ├── XAI_SR.py
│   │   └── SwiftUnet3D.py
│   ├── utilities/
│   │   ├── config.py
│   │   ├── load_data.py
│   │   ├── preprocessing.py
│   │   ├── utils.py
│   │   └── monai_utils.py
│   └── cli.py
├── pyproject.toml
└── setup.py
```

Compatibility wrappers are still available at paths such as `SR_XAI.train`, `SR_XAI.test`, and `SR_XAI.utils`, but new code should import from `SR_XAI.training`, `SR_XAI.inference`, `SR_XAI.models`, and `SR_XAI.utilities`.

## Installation

From `Local_optimizer/`:

```bash
python3 -m pip install -e .
```

Or use the helper script:

```bash
bash install_package.sh
```

This installs the package and the command-line entry point:

```bash
sr-xai --help
```

The full training/inference stack requires PyTorch, MONAI, Quantus, Captum, Nibabel, W&B, and related dependencies listed in `setup.py`.

## Configuration

Default templates are provided in:

- `configs/train_config.yaml`
- `configs/inference_config.yaml`

Important fields:

- `PATH`: output/checkpoint directory
- `PATH1`: root folder containing XAI method folders and input images
- `classifier_checkpoint`: pretrained classifier `.pt` file; can also be set with `SR_XAI_CLASSIFIER_CHECKPOINT`
- `names`: folder names for the nine XAI methods
- `nam`: filename stems/prefixes for the nine XAI maps
- `roi_size`, `height`, `width`, `depth`, `bz`: spatial and batch settings
- `l0`, `l1`, `l2`, `min_com`, `max_faith`: optimizer objective settings

W&B is disabled automatically when `WANDB_API_KEY` is not set. To enable logging:

```bash
export WANDB_API_KEY=...
```

## Run Training

Using the installed CLI:

```bash
sr-xai train --config configs/train_config.yaml
```

Override individual YAML values from the command line:

```bash
sr-xai train --config configs/train_config.yaml --set lr=0.0005 --set bz=2
```

Using the repository script:

```bash
python3 script/train_script.py
```

Using Python:

```python
from SR_XAI.training.runner import train_from_config

train_from_config(
    "configs/train_config.yaml",
    lr=5e-4,
    bz=2,
    classifier_checkpoint="/path/to/resnet18_classifier.pt",
)
```

You can also call the lower-level training function directly:

```python
from SR_XAI.training.train import train

train(
    model="XAI_SwinMT",
    model_name_s="training_sr_xai.pt",
    ep=75,
    lr=5e-3,
    roi_size=64,
    height=64,
    width=64,
    depth=64,
    bz=4,
    PATH="./outputs/",
    PATH1="./data/XAI_QNN/",
    classifier_checkpoint="/path/to/resnet18_classifier.pt",
)
```

## Run Inference

Using the installed CLI:

```bash
sr-xai infer --config configs/inference_config.yaml
```

Override individual values:

```bash
sr-xai infer --config configs/inference_config.yaml --set model_name="'training_sr_xai.pt'" --set save_nii=True
```

Using the repository script:

```bash
python3 script/test_script.py
```

Using Python:

```python
from SR_XAI.inference.runner import inference_from_config

inference_from_config(
    "configs/inference_config.yaml",
    model_name="training_sr_xai.pt",
    classifier_checkpoint="/path/to/resnet18_classifier.pt",
)
```

## Functional Testing

`functional_testing/` contains a lightweight synthetic test that does not require PyTorch or MONAI. It builds a toy 3D Gaussian target, creates nine noisy explanation maps, computes simple faithfulness and complexity metrics, and fuses the explanations.

Run:

```bash
python3 functional_testing/toy_xai_pipeline.py
```

Expected output:

```text
{
  "faithfulness": 0.9629136755999348,
  "complexity": 0.04443359375
}
```

The recorded results are stored in:

- `functional_testing/results/toy_results.json`
- `functional_testing/toy_results.ipynb`

This test is not a replacement for full training/inference; it is a fast smoke test for the optimizer idea and package workflow.
