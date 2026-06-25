# Comparative Evaluation of DL Architectures for Diabetic Retinopathy Classification

> Code for the paper: **"Comparative Evaluation of Deep Learning Architectures and Training Strategies for Diabetic Retinopathy Classification"** — ICPR 2026

## Requirements

- Python 3.10+
- CUDA 12.x

## Installation

```bash
git clone https://github.com/Lamine-MOH/Comparative-Evaluation-of-DL-Architectures-and-Training-Strategies-for-DR-Classification.git
cd Comparative-Evaluation-of-DL-Architectures-and-Training-Strategies-for-DR-Classification
pip install -r requirements.txt
```

## Datasets

All datasets are public and can be downloaded programmatically:

```python
from src.Data import dataset_download, dataset_prepare

# Downloads and prepares any of: "Aptos", "IDRiD", "DDR", "Messidor-2"
path = dataset_download("Aptos")
dataset_path = dataset_prepare("Aptos", path)
```

| Dataset    | Source       | Images |
|------------|--------------|--------|
| Aptos 2019 | Kaggle       | 5,590  |
| IDRiD      | Google Drive | 516    |
| DDR        | Kaggle       | 12,522 |
| Messidor-2 | Kaggle       | 1,748  |

> **Note for IDRiD:** requires no authentication; downloaded via gdown.  
> **Note for Aptos/DDR/Messidor-2:** requires a [Kaggle API token](https://www.kaggle.com/docs/api).

## Running an Experiment

See `experiments/run_experiment.py` for a full configurable pipeline:

```bash
python experiments/run_experiment.py \
  --dataset Aptos \
  --model ResNet50Pretrained \
  --num_classes 5 \
  --epochs 30 \
  --batch_size 16 \
  --img_size 224
```

## Project Structure

```
src/
  Data.py      # Dataset download, preparation, DataLoader
  Model.py     # All model architectures + model_setup factory
  Train.py     # Training loop, validation, checkpointing
  Test.py      # Evaluation, metrics, confusion matrix
experiments/
  run_experiment.py   # CLI entry point
```

## Results

*(Table from paper — e.g., accuracy/AUC per model per dataset)*

## Citation

```bibtex

```

## License

MIT — see [LICENSE](LICENSE).

[Archive on Zenodo](https://doi.org/10.5281/zenodo.20843975)