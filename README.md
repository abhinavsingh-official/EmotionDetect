# EmotionDetect - Voice-Based Emotion Recognition

EmotionDetect is a Python application for identifying emotions from human speech. It uses audio signal processing and machine learning to extract relevant features from voice recordings, compare them to a labeled audio database, and determine emotional similarity scores.

## Features

- Real-time emotion recognition using a microphone
- Extraction of key audio features (MFCC, Mel Spectrogram, Chroma)
- Cosine similarity-based emotion comparison with a reference database
- Bar chart visualization of predicted emotion scores
- Saves emotion results to a CSV file
- Automatically stores visual and result outputs in the `outputs/` directory
- Organized output: top three emotions displayed individually; others grouped

## Folder Structure


EmotionDetect/
- audioemo.py                # Core emotion recognition logic
- record.py                  # Microphone audio recording class
- main.py                    # Entry point for running the system
- audio_db.pkl               # Precomputed emotion feature database
- outputs/                   # Folder for saved bar charts and CSV results
- dataset/                   # (Optional) Folder containing original audio files
- requirements.txt           # Python dependencies
- README.md                  # Project documentation


## Note This requires a Virtual Environment to run.

## Outputs
1.Emotion prediction scores are shown in a bar graph
2.A CSV log is saved to outputs/emotion_results.csv
3.Graph image is saved as outputs/emotion_barchart.png

## How It Works
-Load a preprocessed audio database (Audio_db.pkl)
-Record audio from the user in real time
-Extract features (MFCCs, Mel spectrogram, Chroma)
-Compare user audio features with class-wise averages from the database
-Output similarity scores and visualizations

## Dataset
The project is compatible with the RAVDESS Dataset, which includes emotion-labeled audio files. The emotion database is generated using these samples.