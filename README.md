# Hand Gesture Recognition with XAI (Machine Vision)

This folder contains a complete Kaggle-ready notebook for training a hand gesture recognition CNN with XAI-friendly model outputs.

**Notebook**: The primary file is [mv-project.ipynb](Machine%20Vision/mv-project.ipynb)

**Project goal**: Train a convolutional neural network on the Leap Gesture Recognition dataset to recognize hand gestures, produce evaluation plots, and save a best model for local inference.

## Key features

- Data preprocessing with CLAHE (contrast limited adaptive histogram equalization)
- Augmentation and ImageDataGenerator pipeline
- CNN architecture with a named `last_conv` layer for Grad-CAM / XAI
- Callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
- Validation analysis: confusion matrix, per-class accuracy, confidence analysis, visualizations of correct/incorrect predictions

## Dataset

- Dataset used: Leap Gesture Recognition (add to Kaggle as `gti-upm/leapgestrecog` when running on Kaggle).

## Outputs produced

- `best_gesture_model.h5` (best checkpoint saved during training)
- `gesture_model_final.h5` (final saved model)
- `class_names.json` (mapping of class indices to names)
- PNGs for reports: `sample_images.png`, `training_history.png`, `confusion_matrix.png`, `confidence_distribution.png`, `validation_correct_predictions.png`, `validation_incorrect_predictions.png`, `validation_per_class_accuracy.png`, `validation_confidence_analysis.png`

## Quick start

Running on Kaggle (recommended for training with GPU):

1. Create a new Kaggle notebook and add the dataset: `gti-upm/leapgestrecog`.
2. Enable GPU accelerator (e.g., T4).
3. Copy the notebook content from [mv-project.ipynb](Machine%20Vision/mv-project.ipynb) into the Kaggle notebook and run all cells.

Running locally (for analysis or inference):

1. Install dependencies (create a virtualenv/conda environment first):

```bash
pip install tensorflow opencv-python numpy pandas matplotlib seaborn scikit-learn jupyter
```

2. For local inference only, download `best_gesture_model.h5` and `class_names.json` from the Kaggle output and place them in this folder.

3. Launch a small inference script or open a Jupyter session and load the model:

```python
from tensorflow import keras
model = keras.models.load_model('best_gesture_model.h5')
import json
with open('class_names.json') as f:
    class_map = json.load(f)

# Add your preprocessing (CLAHE + resize) then call model.predict()
```

## Configuration defaults (not exhaustive)

- Image size: 128x128
- Batch size: 64
- Epochs: 40
- Learning rate: 0.001

## Notes & next steps

- The notebook is written to run end-to-end on Kaggle with GPU; running full training locally requires a GPU and sufficient memory.
- If you want, I can extract a small `inference.py` or a `requirements.txt` from the notebook—tell me which you prefer.

---
License: see LICENSE in this folder.
