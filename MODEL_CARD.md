# Model Card

## Project

LaLiga Teams Logo Classifier

## Team

- Tomas Awad
- Ahmed Elserafy
- Mohamed Hesham
- Androw Osama
- Ahmed Mohamed

## Dataset

The dataset is available on Kaggle:

https://www.kaggle.com/datasets/tomasawad/laliga-teams-logo

## Task

Image classification for football club logos.

## Inputs

RGB logo image uploaded by the user.

## Outputs

Top-K class probabilities over 40 team labels.

## Included Models

- Custom CNN
- EfficientNetB0
- DenseNet121
- MobileNetV3Large

## Best Recorded Result

- Model: DenseNet121
- Accuracy: 99.9808%
- Top-5 Accuracy: 100.00%
- Loss: 0.7443

## Limitations

- The classifier performs best on clean, visible football logo images.
- Very low-resolution, cropped, or heavily edited logos may reduce confidence.
- Predictions depend on the training class order in `class_names.json`.
