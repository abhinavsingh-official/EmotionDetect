"""Runs a real pretrained speech-emotion model (ehcalabres/wav2vec2-lg-xlsr-
en-speech-emotion-recognition) for comparison against classifier.py's
from-scratch classical models - no fine-tuning here, just inference, as the
"what does a modern pretrained model get you" reference point.

This needed more than transformers.pipeline("audio-classification", ...):
that loads the checkpoint through AutoModelForAudioClassification, whose
built-in classification head is named projector/classifier (a single linear
layer). This checkpoint's actual head - confirmed by inspecting its
safetensors key names directly - is classifier.dense (1024x1024) followed by
classifier.output (1024x8), the two-layer head from the older wav2vec2
fine-tuning scripts this model was trained with. Loading it through the
generic pipeline silently drops those weights as "unexpected" and
randomly initializes a fresh head instead, which produces near-uniform
~12.5%-per-class output - not a real prediction, just an untrained
classifier. This module loads the wav2vec2 backbone and rebuilds the actual
trained head by hand, by name, from the checkpoint's own weights."""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

MODEL_ID = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
# Confirmed via the checkpoint's own config.json id2label - this is the
# order classifier.output's 8 logits actually correspond to, alphabetical
# because the original training script used an alphabetically sorted label
# set, not RAVDESS's own emotion-code order.
LABELS = ["angry", "calm", "disgust", "fearful", "happy", "neutral", "sad", "surprised"]


class _ClassificationHead(nn.Module):
    """Mirrors the checkpoint's own two-layer head exactly (dense -> tanh ->
    output), just renamed from classifier.dense/classifier.output to plain
    attribute names so state_dict keys line up without a wrapper prefix."""

    def __init__(self, hidden_size: int, num_labels: int):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, num_labels)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.output(torch.tanh(self.dense(pooled)))


class PretrainedEmotionModel:
    def __init__(self):
        self.extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_ID)
        self.backbone = Wav2Vec2Model.from_pretrained(MODEL_ID)
        self.head = _ClassificationHead(self.backbone.config.hidden_size, len(LABELS))

        checkpoint_path = hf_hub_download(MODEL_ID, "model.safetensors")
        state = load_file(checkpoint_path)
        self.head.load_state_dict({
            "dense.weight": state["classifier.dense.weight"],
            "dense.bias": state["classifier.dense.bias"],
            "output.weight": state["classifier.output.weight"],
            "output.bias": state["classifier.output.bias"],
        })

        self.backbone.eval()
        self.head.eval()

    @torch.no_grad()
    def predict(self, audio: np.ndarray, sr: int) -> dict[str, float]:
        inputs = self.extractor(audio, sampling_rate=sr, return_tensors="pt")
        hidden = self.backbone(inputs.input_values).last_hidden_state
        pooled = hidden.mean(dim=1)  # mean-pool over time, the standard wav2vec2-for-classification strategy
        probs = torch.softmax(self.head(pooled), dim=-1).squeeze(0).numpy()
        return dict(zip(LABELS, (float(p) for p in probs)))


if __name__ == "__main__":
    import json

    import librosa
    import pandas as pd
    from tqdm import tqdm

    model = PretrainedEmotionModel()
    df = pd.read_pickle("Audio_db.pkl") if Path("Audio_db.pkl").exists() else None

    import glob

    file_paths = sorted(glob.glob("db/RAVDESS/Actor_*/*.wav"))
    emotion_map = {
        "01": "neutral", "02": "calm", "03": "happy", "04": "sad",
        "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised",
    }

    correct = 0
    total = 0
    for path in tqdm(file_paths, desc="Evaluating pretrained model"):
        true_label = emotion_map[Path(path).name.split("-")[2]]
        audio, sr = librosa.load(path, sr=16000)
        probs = model.predict(audio, sr)
        predicted = max(probs, key=probs.get)
        correct += int(predicted == true_label)
        total += 1

    accuracy = correct / total
    print(f"\nPretrained wav2vec2 accuracy on all {total} RAVDESS samples: {accuracy:.3f}")
    print(
        "Caveat: this model was itself fine-tuned on this exact 1440-sample RAVDESS "
        "set (confirmed via its model card) with no disclosed train/test split, so this "
        "number very likely includes samples the model has already seen during its own "
        "training - it is not a clean generalization test the way the classical models' "
        "cross-validated accuracy is. Reported anyway, with this caveat attached, since "
        "the honest comparison is still informative."
    )

    Path("outputs").mkdir(exist_ok=True)
    with open("outputs/pretrained_results.json", "w") as f:
        json.dump({
            "accuracy": accuracy,
            "total_samples": total,
            "caveat": "Evaluated on the model's own likely training set (RAVDESS, no disclosed split) - not a clean generalization measurement.",
        }, f, indent=2)
