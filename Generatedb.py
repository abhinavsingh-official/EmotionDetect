import glob
from AudioEmo import AudioEmotionRecognizer

path_list = glob.glob("db/RAVDESS/Actor_*/*.wav")
print(f"Total files found: {len(path_list)}")

emotion_map = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

df = AudioEmotionRecognizer.create_database(path_list, emotion_index=2, emotion_map=emotion_map)
    