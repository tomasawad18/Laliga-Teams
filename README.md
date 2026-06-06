# LaLiga Teams Logo Classifier

A deep learning web app that classifies Spanish football team logos from an uploaded image.

This project was built as a team project by:

- Tomas Awad
- Ahmed Elserafy
- Mohamed Hesham
- Androw Osama
- Ahmed Mohamed

The project includes a Gradio interface and four trained Keras/TensorFlow models:

- Custom CNN
- EfficientNetB0
- DenseNet121
- MobileNetV3Large

The app can run a single-model prediction or compare the predictions of all four models.

## Dataset

The training dataset is available on Kaggle:

**LaLiga Teams Logo Dataset**  
https://www.kaggle.com/datasets/tomasawad/laliga-teams-logo

Use the Kaggle dataset to reproduce training, experiment with new architectures, or expand the project.

## Demo Features

- Upload any club logo image.
- Optional logo background cleanup using OpenCV GrabCut.
- Resize with padding to avoid logo distortion.
- Top-K prediction probabilities.
- Compare all trained models in one screen.
- Deployment-ready Gradio app.

## Classes

The model predicts **40 classes**:

Albacete, Almeria, Andorra, Athletic Bilbao, Atletico Madrid, Barcelona, Burgos, Cadiz, Castellon, Celta Vigo, Ceuta, Cultural y Deportiva Leonesa, Deportivo Alaves, Deportivo de La Coruna, Eibar, Elche, Espanyol, Getafe, Girona, Granada, Huesca, Las Palmas, Leganes, Levante, Malaga, Mallorca, Mirandes, Osasuna, Rayo Vallecano, Real Betis, Real Madrid, Real Oviedo, Real Racing, Real Sociedad, Real Valladolid, Real Zaragoza, Sevilla, Sporting Gijon, Valencia, Villarreal

## Best Recorded Result

From the included training results file:

| Model | Accuracy | Top-5 Accuracy | Loss | Finished At |
|---|---:|---:|---:|---|
| DenseNet121 | 99.9808% | 100.00% | 0.7443 | 2026-06-04T20:21:58 |

## Project Structure

```text
laliga-logo-classifier-github-ready/
├── app.py
├── class_names.json
├── requirements.txt
├── runtime.txt
├── README.md
├── LINKEDIN_POST.md
├── MODEL_CARD.md
├── DATASET.md
├── .gitignore
├── models/
│   ├── best_laliga_logo_cnn.keras
│   ├── best_laliga_logo_densenet121.h5
│   ├── best_laliga_logo_efficientnet.keras
│   └── best_laliga_logo_mobilenetv3_large.keras
├── notebooks/
│   └── laliga_logo_training.ipynb
└── reports/
    └── all_results.json
```

## Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/laliga-logo-classifier.git
cd laliga-logo-classifier
```

### 2. Create and activate a virtual environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install requirements

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run the app

```bash
python app.py
```

Then open the local Gradio URL shown in the terminal.

## Upload to GitHub

Because model files are included, use Git from the terminal instead of the GitHub website upload.

```bash
git init
git add .
git commit -m "Add LaLiga logo classifier Gradio app"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/laliga-logo-classifier.git
git push -u origin main
```

If GitHub warns about large files, install and use Git LFS:

```bash
git lfs install
git lfs track "*.keras"
git lfs track "*.h5"
git add .gitattributes models/
git commit -m "Track model files with Git LFS"
git push
```

## Deployment Notes

For Render, Hugging Face Spaces, or similar platforms:

- Start command: `python app.py`
- Python version: `3.11`
- The app reads the `PORT` environment variable automatically.
- The app launches on `0.0.0.0`, which is required for most deployment platforms.

## Important Notes

- Keep `class_names.json` in the exact same class order used during training.
- The app assumes the models already contain their own preprocessing/rescaling layer.
- If a `.keras` model was saved using a newer Keras version, `app.py` creates a temporary compatible copy inside `models/_compatible/`.

## Team Credits

Developed by Tomas Awad, Ahmed Elserafy, Mohamed Hesham, Androw Osama, and Ahmed Mohamed.
