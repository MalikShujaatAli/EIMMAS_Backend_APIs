# Phase 1 Thesis: CNN Model Training & GPU Diagnostics

## Strategic Intent
The hypothesis of Phase 1 was binary: "Can a Convolutional Neural Network learn to distinguish between seven human facial emotions from grayscale images?" This was a feasibility probe — no API, no deployment, no integration. The sole success criterion was achieving a classification accuracy materially above random chance (14.3% for 7 classes).

## Scope & Boundaries
Phase 1 was confined to two activities: (1) verifying that the development machine had GPU access (`2.py`), and (2) training a shallow 3-layer CNN on the FER2013 dataset at 48×48 pixel resolution (`1.py`). The dataset was stored locally in `archive_5/`. The output artifact was `emotion_model.h5`, later renamed/retrained as `face_emotion_model.h5`. A Jupyter notebook (`cnn model/1.ipynb`) was used for the actual training run, producing diagnostic plots and a classification report.

## Failure Analysis
Phase 1 ended because the model achieved only **57% weighted accuracy**. While this exceeded random chance, the per-class breakdown was catastrophic for minority emotions: Disgust achieved 3% recall (effectively invisible to the model), Fear achieved 18% recall, and Angry achieved only 51% recall. The root causes were threefold: (1) the FER2013 dataset contains notoriously noisy labels (multiple humans disagree on the "correct" emotion for many images), (2) the 3-layer CNN architecture was too shallow to capture the geometric complexity of facial expressions, and (3) no data augmentation, class weighting, or lighting normalization (CLAHE) was applied. These deficiencies were not addressed until Phase 7 (2nd attempt Video) and the Kaggle retraining documented in `Model notebooks.txt`, which upgraded to a 4-block CNN with BatchNormalization, CLAHE preprocessing, 112×112 resolution, the FERPlus dataset (Microsoft-corrected labels), and class weighting — achieving **81.03% accuracy**.
