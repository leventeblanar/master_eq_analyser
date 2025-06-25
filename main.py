import librosa
import numpy as np
import matplotlib.pyplot as plt
import os

file_path = "Inhumanity_solo_14.wav"

if not os.path.exists(file_path):
    raise FileNotFoundError(f"A fájl nem található: {file_path}")

y, sr = librosa.load(file_path, sr=None)

fft = np.fft.fft(y)

magnitude = np.abs(fft)

frequency = np.fft.fftfreq(len(magnitude), 1/sr)

half_freq = frequency[]