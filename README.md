# EmotionDetect

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white) ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white) ![scikit--learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white) ![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black) ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)

An emotion timeline for audio. Upload a film scene, a song, or a recorded
conversation and see how the emotion in it changes over time, in two
modes: general clips and an interview/interrogation mode with an
on-screen disclaimer that tone is a signal, not lie detection. Under the
hood, a sliding window runs the same classifier across the clip -
features are 180 numbers per window (Mel-spectrogram, MFCCs, chroma), fed
through an MLP selected by 5-fold cross-validation against an SVM, a
Random Forest, and a pretrained wav2vec2 model (Model Comparison tab, with
a full confusion matrix and permutation-importance explainability). Live
microphone recording is supported directly in the browser, and everything
runs locally with no API key.

## Screenshots

![Home tab](screenshots/home.png)
![Timeline tab](screenshots/timeline.png)
![Timeline interrogation mode](screenshots/timeline_interrogation.png)
![Model Comparison tab](screenshots/model_comparison.png)

## How to run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Get the RAVDESS dataset and build the models:

```bash
mkdir -p db && cd db
curl -L -o ravdess_speech.zip "https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip"
mkdir RAVDESS && unzip ravdess_speech.zip -d RAVDESS && rm ravdess_speech.zip
cd ..
python3 Generatedb.py
python3 classifier.py
python3 explainability.py
python3 pretrained.py
```

```bash
streamlit run app.py
```
