# Sanskrit Character Recognition using CNN

A Convolutional Neural Network built in Python to recognize handwritten Sanskrit characters
segmented from real Sanskrit documents.

## Dataset

**Sanskrit Letter Dataset** — created by the DevDigitizer project for Sanskrit OCR research.

- Source: https://github.com/avadesh02/Sanskrit-letter-dataset
- 7,702 images of Sanskrit (Devanagari) letters, 602 classes
- Images are 32×32 RGB, labelled with I-Trans transliteration (`ka`, `va`, `ti`, `bha`, ...)
- Stored as a single pickle file `dev_letter_D.p` (77 MB)

**Subset used in this project:** the 20 most frequent letter classes, 25 images each
= **500 images** (400 training / 100 testing).

Citation:

> Avadesh, Meduri, and Navneet Goyal. "Optical Character Recognition for Sanskrit Using
> Convolution Neural Networks." *13th IAPR International Workshop on Document Analysis
> Systems (DAS)*, 2018.

## Files

| File | Description |
|---|---|
| `Sanskrit_Character_CNN.ipynb` | Main notebook (Google Colab ready) |
| `sanskrit_cnn.py` | Same pipeline as a standalone script |
| `requirements.txt` | Dependencies |

## Method

1. Download and load the pickle dataset
2. Select the 20 most frequent letter classes, 25 images each
3. Convert to grayscale with OpenCV and resize to 32×32
4. Split 80/20 into train/test (stratified), normalize pixels to 0–1
5. Encode labels with `LabelEncoder` and one-hot encoding
6. Train a CNN: `Conv2D(16) → MaxPool → Conv2D(32) → MaxPool → Dense(64) → Dropout(0.4) → Softmax(20)`
7. Evaluate with accuracy, precision, recall, F1-score and a confusion matrix
8. Plot training vs validation curves and visualize predictions

## Results

Measured test accuracy: **0.82**

| Metric | Score |
|---|---|
| Accuracy | 0.82 |
| Macro avg precision | 0.86 |
| Macro avg recall | 0.82 |
| Macro avg F1-score | 0.81 |
| Weighted avg precision | 0.86 |
| Weighted avg recall | 0.82 |
| Weighted avg F1-score | 0.81 |

Test set: 100 images, 5 per class.

Accuracy varies roughly between **0.76 and 0.84** across runs. With only five test
images per class, a single image is worth a full percentage point, and weight
initialisation and dropout add further randomness — so re-running will not reproduce
0.82 exactly.

Ten of the twenty classes were recognised perfectly (`H`, `a`, `bha`, `i`, `ka`,
`m`, `ra`, `ta`, `ti`, `vaa`). The 18 errors were spread across the other ten
classes rather than forming one systematic confusion.

Most frequent errors:

| True | Predicted | Count |
|---|---|---|
| `ya` | `pa` | 3 |
| `va` | `m` | 2 |
| others | various | 1 each |

The weakest class is `ya` (recall 0.20) — only one of its five test images was
classified correctly, with three going to `pa`. The lowest precision is `pa` (0.44),
which absorbed wrong predictions from `ya`, `n` and `sa`. The `ya`/`pa` confusion is
visually plausible: both glyphs are built around a closed left loop joined to a
vertical stem.

## How to run

**Google Colab:** open the notebook and run all cells. The first cell downloads the dataset.

**Locally:**

```bash
pip install -r requirements.txt
wget https://raw.githubusercontent.com/avadesh02/Sanskrit-letter-dataset/master/dev_letter_D.p
python sanskrit_cnn.py
```

## Tuning the accuracy

Settings at the top of the file control the difficulty:

```python
NUM_CLASSES      = 20
IMAGES_PER_CLASS = 25
EPOCHS           = 30
```

Measured results for other settings:

| Classes | Images/class | Epochs | Accuracy |
|---|---|---|---|
| 20 | 25 | 20 | ~0.66 |
| 20 | 25 | 30 | ~0.76 - 0.84 |
| 20 | 25 | 40 | ~0.84 |
| 10 | all (~180) | 20 | ~0.93 |
| 40 | 30 | 20 | ~0.80 |

- **Higher accuracy:** raise `IMAGES_PER_CLASS` and `EPOCHS`, or lower `NUM_CLASSES`
- **Lower accuracy:** the reverse
