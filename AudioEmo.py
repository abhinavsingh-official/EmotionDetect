import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.signal import resample
from sklearn.preprocessing import RobustScaler
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict


class AudioEmotionRecognizer:
    def __init__(self, db_path: str, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.df = pd.read_pickle(db_path)
        self.features = [col for col in self.df.columns if col != 'target']
        self.scaler = RobustScaler().fit(self.df[self.features])
        self.data = self.scaler.transform(self.df[self.features])
        self.labels = self.df['target']
        self.class_means = self._compute_means()

    def _compute_means(self) -> Dict[str, np.ndarray]:
        df_scaled = pd.DataFrame(self.data, columns=self.features)
        df_scaled['target'] = self.labels
        return {
            label: df_scaled[df_scaled['target'] == label][self.features]
                   .mean().values.reshape(1, -1)
            for label in df_scaled['target'].unique()
        }

    def _extract_features(self, audio: np.ndarray, sr: int) -> np.ndarray:
        
        if sr != self.sample_rate:
            audio = resample(audio, int(len(audio) * self.sample_rate / sr))

        # Convert audio to float32 in range [-1, 1]
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32) / np.max(np.abs(audio))

        mel = librosa.feature.melspectrogram(y=audio, sr=self.sample_rate)
        chroma = librosa.feature.chroma_stft(y=audio, sr=self.sample_rate)
        mfcc = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=40)

        combined = np.concatenate([mel, mfcc, chroma], axis=0)
        features = combined.mean(axis=1).reshape(1, -1)
        return self.scaler.transform(features)

    def predict(self, audio: np.ndarray, sr: int) -> Dict[str, float]:

        features = self._extract_features(audio, sr)
        return {
            label: float(cosine_similarity(mean, features)[0, 0])
            for label, mean in self.class_means.items()
        }

    @staticmethod
    def create_database(
        file_paths: List[str],
        emotion_index: int,
        emotion_map: Dict[str, str],
        sample_rate: int = 16000,
        n_mfcc: int = 40
    ) -> pd.DataFrame:
    
        def extract(audio, sr):
            mel = librosa.feature.melspectrogram(y=audio, sr=sr)
            chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
            combined = np.concatenate([mel, mfcc, chroma], axis=0)
            return combined.mean(axis=1)

        features, labels = [], []

        for path in tqdm(file_paths, desc="Building DB"):
            audio, _ = librosa.load(path, sr=sample_rate)
            emotion_id = path.split("\\")[-1].split("-")[emotion_index]
            label = emotion_map.get(emotion_id, "Unknown")
            features.append(extract(audio, sample_rate))
            labels.append(label)

        df = pd.DataFrame(features, columns=[f'f{i}' for i in range(features[0].shape[0])])
        df['target'] = labels
        df.to_pickle(f'Audio_db.pkl')
        print("Audio database saved.")
        return df
