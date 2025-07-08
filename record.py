import pyaudio
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
import threading


class Recorder:
    def __init__(self, chunk=4000, sr=16000, wait=0.5, dtype='int16', channels=1, threshold=None):
        self.chunk = chunk
        self.sr = sr
        self.wait = wait
        self.dtype = dtype
        self.channels = channels
        self.chunks_per_sec = sr // chunk
        self.format = self._get_format(dtype)
        self.threshold = threshold if threshold is not None else self._default_threshold(dtype)

        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(rate=sr, channels=channels, format=self.format,
                                      input=True, output=False, frames_per_buffer=chunk)

    def _get_format(self, dtype):
        return {
            'int16': pyaudio.paInt16,
            'int32': pyaudio.paInt32,
            'float32': pyaudio.paFloat32
        }[dtype]

    def _default_threshold(self, dtype):
        return {'int16': 2**13, 'int32': 2**26, 'float32': 0.1}[dtype]

    def _should_stop(self, buffer):
        tail = int(self.wait * self.sr)
        return buffer[-tail:].max() < self.threshold

    def print_audio_devices(self):
        for i in range(self.audio.get_device_count()):
            print(self.audio.get_device_info_by_index(i)['name'])

    def record(self):
        print("Press Enter to start recording...")
        input()

        print("Recording... Press Enter to stop.")
        frames = []
        stop_flag = False

        def wait_for_enter():
            nonlocal stop_flag
            input()
            stop_flag = True

        thread = threading.Thread(target=wait_for_enter)
        thread.start()

        while not stop_flag:
            buffer = self.stream.read(self.chunk)
            audio = np.frombuffer(buffer, self.dtype)
            frames.append(audio)
            print(f"Volume: {audio.max():>6}", end='\r')

        print("\nRecording stopped.")
        data = np.concatenate(frames)
        return data, b''.join([f.tobytes() for f in frames])

    def record_and_visualize(self):
        plt.ion()
        fig, ax = plt.subplots()
        line, = ax.plot(np.zeros(self.chunk))

        ax.set_ylim({
            'int16': (-2**15, 2**15 - 1),
            'int32': (-2**31, 2**31 - 1),
            'float32': (-1.0, 1.0)
        }[self.dtype])

        print("Listening with visualization...")
        while True:
            raw = self.stream.read(self.chunk)
            audio = np.frombuffer(raw, self.dtype)
            line.set_ydata(audio)
            fig.canvas.draw()
            fig.canvas.flush_events()

            if audio.max() > self.threshold:
                print("Recording started")
                buffer = [audio]
                raw_data = raw

                for _ in range(int(self.wait * self.chunks_per_sec * 3)):
                    raw = self.stream.read(self.chunk)
                    audio = np.frombuffer(raw, self.dtype)
                    buffer.append(audio)
                    raw_data += raw

                    line.set_ydata(audio)
                    fig.canvas.draw()
                    fig.canvas.flush_events()

                    if self._should_stop(np.concatenate(buffer)):
                        print("Recording stopped")
                        plt.close()
                        break

                return np.concatenate(buffer), raw_data

    @staticmethod
    def write_to_wav(filename, sr, data):
        wavfile.write(filename, sr, data)
        print(f"Saved recording as {filename}")
