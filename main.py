import os
import csv
from datetime import datetime
import matplotlib.pyplot as plt

from record import Recorder
from AudioEmo import AudioEmotionRecognizer


def plot_emotion_bar_chart(results: dict):
    os.makedirs("outputs", exist_ok=True)
    emotions = list(results.keys())
    scores = list(results.values())

    plt.figure(figsize=(10, 5))
    bars = plt.bar(emotions, scores, color='skyblue')
    plt.ylabel("Similarity Score")
    plt.xlabel("Emotion")
    plt.title("Detected Emotion Similarity Scores")

    for bar, score in zip(bars, scores):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{score:.2f}', ha='center', va='bottom')

    chart_path = os.path.join("outputs", "emotion_barchart.png")
    plt.savefig(chart_path)
    plt.close()
    print(f"Bar chart saved to: {chart_path}")


def save_results_to_csv(results: dict, filename="emotion_results.csv"):
    os.makedirs("outputs", exist_ok=True)
    filepath = os.path.join("outputs", filename)
    file_exists = os.path.isfile(filepath)

    with open(filepath, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(['Timestamp'] + list(results.keys()))
        writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S")] + list(results.values()))

    print(f"CSV saved to: {filepath}")


if __name__ == "__main__":
    recognizer = AudioEmotionRecognizer("Audio_db.pkl", sample_rate=16000)
    recorder = Recorder(sr=16000, threshold=2000)
    print("Speak into the microphone...")
    audio, _ = recorder.record()
    results = recognizer.predict(audio, sr=16000)

    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    print("\nTop 3 Detected Emotions:")
    for emotion, score in sorted_results[:3]:
        print(f"{emotion}: {round(score, 4)}")

    if len(sorted_results) > 3:
        print("\nOther Emotions:")
        for emotion, score in sorted_results[3:]:
            print(f"{emotion}: {round(score, 4)}")

    plot_emotion_bar_chart(results)
    save_results_to_csv(results)
