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

### Results of Experiment 1
*Replication of Literature Methods Across Multiple Datasets*

| Datasets | Methods | Accuracy (Binary) | Precision (Binary) | Recall (Binary) | F1-Score (Binary) | Specificity (Binary) | Accuracy (Multi) | Precision (Multi) | Recall (Multi) | F1-Score (Multi) | Specificity (Multi) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **Aptos** | Modified Xception | _0.951_ | _0.9511_ | _0.9515_ | _0.951_ | _0.9464_ | 0.6921 | 0.485 | 0.3747 | 0.328 | 0.913 |
| | Hybrid CNN-SVD with ELM | 0.9277 | 0.9468 | 0.9086 | 0.9273 | 0.9224 | _0.7299_ | _0.6961_ | **0.7299** | **0.7091** | _0.9237_ |
| | Hybrid MobileNetV2 + SVM | **0.9754** | **0.9755** | **0.9754** | **0.9754** | **0.9754** | **0.8022** | **0.6632** | _0.6129_ | _0.6287_ | **0.9488** |
| | EfficientNet-B5 and CLAHE | 0.7763 | 0.7849 | 0.7935 | 0.7756 | 0.7011 | 0.6685 | 0.2654 | 0.3486 | 0.2963 | 0.8991 |
| | VGG-16 with novel Preprocessing | 0.9222 | 0.9223 | 0.9222 | 0.9222 | 0.9222 | 0.6944 | 0.3494 | 0.403 | 0.3727 | 0.9139 |
| | VGG16 | 0.9059 | 0.9058 | 0.9059 | 0.9058 | 0.9059 | 0.7067 | 0.3371 | 0.3775 | 0.3407 | 0.9089 |
| | InceptionV3 | 0.8799 | 0.8877 | 0.881 | 0.8795 | 0.881 | 0.6739 | 0.4075 | 0.3735 | 0.3549 | 0.8979 |
| **IDRiD** | Modified Xception | 0.769 | 0.754 | 0.692 | 0.706 | 0.692 | 0.4 | 0.133 | 0.333 | 0.19 | 0.667 |
| | Hybrid CNN-SVD with ELM | 0.6442 | 0.7089 | **0.8** | 0.7517 | 0.5832 | 0.4038 | **0.4257** | **0.4038** | **0.399** | 0.8268 |
| | Hybrid MobileNetV2 + SVM | **0.8173** | **0.7932** | _0.7887_ | **0.7908** | **0.7887** | 0.4712 | _0.3359_ | 0.333 | _0.3286_ | 0.8515 |
| | EfficientNet-B5 and CLAHE | 0.625 | 0.3125 | 0.5 | 0.3846 | 0 | 0.4135 | 0.1698 | 0.2529 | 0.2017 | 0.8257 |
| | VGG-16 with novel Preprocessing | 0.7404 | 0.7375 | 0.6332 | 0.6396 | 0.6332 | 0.4615 | 0.3176 | 0.287 | 0.2411 | 0.2411 |
| | VGG16 | _0.7788_ | _0.7587_ | 0.7147 | 0.7279 | 0.7147 | **0.5192** | 0.2141 | 0.3176 | 0.2539 | **0.8571** |
| | InceptionV3 | 0.7692 | 0.7485 | 0.7756 | _0.7538_ | _0.7756_ | _0.5_ | 0.2943 | _0.3337_ | 0.3083 | _0.8568_ |
| **DDR** | Modified Xception | _0.798_ | 0.801 | _0.798_ | _0.798_ | 0.798 | 0.652 | 0.361 | 0.309 | 0.301 | 0.875 |
| | Hybrid CNN-SVD with ELM | 0.7992 | _0.8101_ | 0.781 | 0.7953 | 0.7868 | _0.6766_ | **0.6352** | **0.6766** | **0.651** | _0.8943_ |
| | Hybrid MobileNetV2 + SVM | **0.8335** | **0.8338** | **0.8335** | **0.8335** | _0.8335_ | **0.7788** | _0.602_ | _0.5161_ | _0.5391_ | **0.9239** |
| | EfficientNet-B5 and CLAHE | 0.6655 | 0.6837 | 0.6426 | 0.6345 | **0.8687** | 0.5653 | 0.2206 | 0.2571 | 0.2374 | 0.845 |
| | VGG-16 with novel Preprocessing | 0.6643 | 0.7074 | 0.6645 | 0.6461 | 0.6645 | 0.5485 | 0.2781 | 0.2717 | 0.2548 | 0.8516 |
| | VGG16 | 0.7042 | 0.7081 | 0.7041 | 0.7028 | 0.7041 | 0.5996 | 0.3479 | 0.2725 | 0.2608 | 0.8502 |
| | InceptionV3 | 0.7545 | 0.7546 | 0.7545 | 0.7545 | 0.7545 | 0.6551 | 0.5899 | 0.3394 | 0.3471 | 0.8785 |
| **Messidor2** | Modified Xception | 0.491 | 0.554 | 0.539 | 0.471 | 0.539 | 0.583 | 0.117 | 0.2 | 0.147 | 0.8 |
| | Hybrid CNN-SVD with ELM | 0.6648 | 0.6148 | 0.5172 | 0.5618 | 0.6371 | 0.5559 | 0.4815 | _0.5559_ | _0.5031_ | _0.8248_ |
| | Hybrid MobileNetV2 + SVM | **0.7994** | **0.7985** | **0.7846** | **0.7892** | _0.7846_ | **0.7421** | **0.7345** | **0.5682** | **0.6085** | **0.9135** |
| | EfficientNet-B5 and CLAHE | _0.7393_ | 0.3696 | 0.5 | 0.425 | **1** | 0.5817 | 0.1167 | 0.199 | 0.1471 | 0.7993 |
| | VGG-16 with novel Preprocessing | 0.6619 | 0.6823 | 0.6111 | 0.5959 | 0.6111 | 0.5845 | 0.1169 | 0.2 | 0.1476 | 0.8 |
| | VGG16 | 0.7106 | 0.7036 | 0.6906 | 0.6937 | 0.6906 | 0.5817 | 0.18 | 0.2067 | 0.1669 | 0.8046 |
| | InceptionV3 | 0.7278 | _0.7266_ | _0.7033_ | _0.7075_ | 0.7033 | _0.6361_ | _0.6825_ | 0.3632 | 0.3963 | 0.8404 |


### Results of Experiment 2
*Ablation Study on Training from Scratch vs. Pretrained Models*

| Dataset | Model | Acc (Scratch, Binary) | Prec (Scratch, Binary) | Rec (Scratch, Binary) | F1 (Scratch, Binary) | Acc (Pretrained, Binary) | Prec (Pretrained, Binary) | Rec (Pretrained, Binary) | F1 (Pretrained, Binary) | Acc (Scratch, Multi) | Prec (Scratch, Multi) | Rec (Scratch, Multi) | F1 (Scratch, Multi) | Acc (Pretrained, Multi) | Prec (Pretrained, Multi) | Rec (Pretrained, Multi) | F1 (Pretrained, Multi) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Aptos** | ResNet50 | _0.9418_ | _0.942_ | _0.942_ | _0.9418_ | 0.9655 | _0.9662_ | _0.9658_ | _0.9655_ | _0.7509_ | _0.4177_ | _0.4156_ | _0.4156_ | **0.82** | **0.6825** | **0.666** | **0.6605** |
| | VGG16 | 0.94 | 0.94 | 0.94 | 0.94 | 0.95 | 0.95 | 0.95 | 0.95 | 0.72 | 0.28 | 0.37 | 0.32 | 0.7673 | 0.59 | 0.53 | 0.54 |
| | Inception | 0.51 | 0.25 | 0.5 | 0.34 | **0.97** | **0.97** | **0.97** | **0.97** | 0.723 | 0.29 | 0.37 | 0.32 | _0.7923_ | _0.62_ | _0.59_ | _0.6_ |
| | AlexNet | **0.95** | **0.95** | **0.95** | **0.95** | _0.9673_ | **0.97** | **0.97** | **0.97** | **0.76** | **0.6** | **0.49** | **0.5** | 0.7545 | 0.52 | 0.48 | 0.48 |
| **IDRiD** | ResNet50 | 0.3205 | 0.1603 | 0.5 | 0.2427 | 0.6795 | 0.3397 | 0.5 | 0.4046 | 0.3205 | 0.0641 | 0.2 | 0.0971 | 0.3205 | 0.0641 | 0.2 | 0.0971 |
| | VGG16 | 0.6282 | 0.58 | 0.58 | 0.58 | 0.7436 | 0.71 | 0.66 | 0.67 | **0.4872** | **0.2** | **0.3** | **0.24** | **0.5** | 0.22 | _0.31_ | _0.25_ |
| | Inception | _0.65_ | _0.62_ | _0.63_ | 0.62 | **0.82** | **0.79** | **0.79** | **0.79** | _0.42_ | _0.18_ | _0.26_ | _0.21_ | _0.47_ | **0.29** | _0.31_ | **0.29** |
| | AlexNet | **0.6923** | **0.84** | **0.52** | **0.45** | _0.7692_ | _0.76_ | _0.68_ | _0.7_ | 0.3205 | 0.06 | 0.2 | 0.1 | **0.5** | _0.28_ | **0.32** | **0.29** |
| **DDR** | ResNet50 | **0.8164** | **0.8191** | **0.8164** | **0.816** | **0.885** | **0.8912** | **0.885** | **0.8846** | 0.6317 | 0.2471 | 0.2832 | 0.2619 | **0.8084** | _0.6793_ | **0.5463** | **0.5846** |
| | VGG16 | 0.5003 | 0.25 | 0.5 | 0.33 | 0.8159 | 0.82 | 0.82 | 0.82 | **0.76** | **0.56** | **0.45** | **0.47** | 0.7709 | **0.69** | _0.53_ | _0.54_ |
| | Inception | 0.5 | 0.25 | 0.5 | 0.33 | 0.85 | _0.85_ | _0.85_ | _0.85_ | _0.6824_ | 0.38 | _0.33_ | _0.33_ | _0.7748_ | **0.69** | _0.53_ | _0.54_ |
| | AlexNet | _0.79_ | _0.8_ | _0.79_ | _0.79_ | 0.8233 | 0.82 | 0.82 | 0.82 | 0.6668 | _0.41_ | 0.31 | 0.31 | 0.7174 | 0.39 | 0.41 | 0.4 |
| **Messidor-2** | ResNet50 | 0.584 | 0.292 | _0.5_ | 0.3687 | 0.584 | 0.292 | 0.5 | 0.3687 | 0.585 | 0.1168 | 0.2 | 0.1475 | 0.585 | 0.1168 | 0.2 | 0.1475 |
| | VGG16 | _0.6985_ | _0.69_ | **0.7** | _0.69_ | 0.7366 | _0.73_ | _0.72_ | _0.72_ | _0.6069_ | **0.21** | _0.24_ | _0.21_ | _0.5992_ | _0.32_ | _0.22_ | 0.18 |
| | Inception | 0.58 | 0.29 | _0.5_ | 0.37 | **0.74** | **0.74** | **0.73** | **0.73** | **0.61** | **0.21** | **0.25** | **0.22** | **0.64** | **0.45** | **0.39** | **0.41** |
| | AlexNet | **0.7137** | **0.71** | **0.7** | **0.7** | 0.626 | 0.64 | 0.57 | 0.53 | 0.5802 | _0.19_ | 0.21 | 0.18 | 0.5802 | 0.19 | _0.22_ | _0.2_ |


## Citation

```bibtex

```

## License

MIT — see [LICENSE](LICENSE).

[Archive on Zenodo](https://doi.org/10.5281/zenodo.20843975)