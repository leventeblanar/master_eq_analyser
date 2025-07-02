import streamlit as st
import numpy as np
import librosa
import matplotlib.pyplot as plt
import io

st.title("🎧 Spektrum Analyzer")

uploaded_file = st.file_uploader("WAV fájl feltöltése", type=["wav"])
bin_size = st.slider("Bin méret (simítás mértéke)", min_value=10, max_value=1000, value=100, step=10)

def process_audio(file, bin_size):
    y, sr = librosa.load(file, sr=None)
    fft = np.fft.fft(y)
    magnitude = np.abs(fft)
    freq = np.fft.fftfreq(len(magnitude), 1 / sr)

    half_len = len(magnitude) // 2
    freq = freq[:half_len]
    magnitude = magnitude[:half_len]

    # Bin spektrum
    binned_freq = []
    binned_mag = []
    for i in range(0, len(freq), bin_size):
        f = freq[i:i+bin_size]
        m = magnitude[i:i+bin_size]
        if len(f) > 0:
            binned_freq.append(np.mean(f))
            binned_mag.append(np.mean(20 * np.log10(m + 1e-10)))
    return binned_freq, binned_mag

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")
    freq, mag = process_audio(uploaded_file, bin_size)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(freq, mag)
    ax.set_title("Átlagolt FrekvenciaSpektrum")
    ax.set_xlabel("Frekvencia (Hz)")
    ax.set_ylabel("Magnitúdó (dB)")
    ax.grid(True)
    st.pyplot(fig)
