import librosa
import numpy as np
import matplotlib.pyplot as plt
import os

class AudioAnalyzer:
    def __init__(self, file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"A fájl nem található.")
        self.file_path = file_path
        self.y, self.sr = librosa.load(file_path, sr=None)
        self.fft = None
        self.magnitude = None
        self.frequency = None

    def compute_fft(self):
        self.fft = np.fft.fft(self.y)
        self.magnitude = np.abs(self.fft)
        self.frequency = np.fft.fftfreq(len(self.magnitude), 1 / self.sr)

    def get_half_spectrum(self):
        half_len = len(self.magnitude) // 2
        return self.frequency[:half_len], self.magnitude[:half_len]
    
class Visualizer:
    @staticmethod
    def bin_spectrum(freq, magnitude, bin_size=100):
        binned_freq = []
        binned_mag = []

        for i in range(0, len(freq), bin_size):
            bin_f = freq[i:i+bin_size]
            bin_m = magnitude[i:i+bin_size]

            if len(bin_f) == 0:
                continue

            binned_freq.append(np.mean(bin_f))
            binned_mag.append(np.mean(20 * np.log10(bin_m + 1e-10)))

        return binned_freq, binned_mag


    @staticmethod
    def plot_binned_spectrum(freq, magnitude, bin_size= 100, title="FreqvenciaSpektrum"):
        binned_freq, binned_mag = Visualizer.bin_spectrum(freq, magnitude, bin_size)

        plt.figure(figsize=(10,6))
        plt.plot(binned_freq, binned_mag)
        plt.xlabel("Freqvencia")
        plt.ylabel("Magnitudó")
        plt.title(title)
        plt.grid(True)
        plt.xlim(0, 20000)
        plt.tight_layout()
        plt.show()

def main():
    file_path = "Inhumanity_solo_14.wav"

    analyzer = AudioAnalyzer(file_path)
    analyzer.compute_fft()
    freq, mag = analyzer.get_half_spectrum()

    Visualizer.plot_binned_spectrum(freq, mag)

if __name__ == '__main__':
    main()