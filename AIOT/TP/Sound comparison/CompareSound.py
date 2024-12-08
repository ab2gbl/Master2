import os
import time
import numpy as np
import librosa
from PyQt5 import QtWidgets, QtGui
import sounddevice as sd
from scipy.io.wavfile import write
import soundfile as sf
from dataset import SessionLocal
from models import OriginalSound, ComparisonSound
from sqlalchemy import func
import matplotlib.pyplot as plt
import librosa.display
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIO_FOLDER = os.path.join(BASE_DIR, "compared_sounds")
SOUNDS_FOLDER = os.path.join(AUDIO_FOLDER, "sounds")
SPECTROGRAMS_FOLDER = os.path.join(AUDIO_FOLDER, "spectrograms")
WAVEFORMS_FOLDER = os.path.join(AUDIO_FOLDER, "waveforms")

# Create the directories if they don't exist
for folder in [SOUNDS_FOLDER, SPECTROGRAMS_FOLDER, WAVEFORMS_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

class App2CompareSound(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sound Comparison App")
        self.setGeometry(100, 100, 800, 600)

        # GUI Elements
        self.record_button = QtWidgets.QPushButton("Record Current Sound", self)
        self.record_button.setGeometry(200, 50, 200, 50)
        self.record_button.clicked.connect(self.recordCurrentAudio)

        self.play_button = QtWidgets.QPushButton("Play Current Sound", self)
        self.play_button.setGeometry(200, 120, 200, 50)
        self.play_button.clicked.connect(self.playCurrentAudio)

        self.spectrogram_button = QtWidgets.QPushButton("Display Spectrogram", self)
        self.spectrogram_button.setGeometry(200, 190, 200, 50)
        self.spectrogram_button.clicked.connect(self.showSpectrogram)
        
        self.compare_button = QtWidgets.QPushButton("Compare Sounds", self)
        self.compare_button.setGeometry(200, 260, 200, 50)
        self.compare_button.clicked.connect(self.compareAudio)

        self.status_label = QtWidgets.QLabel("Status: Ready", self)
        self.status_label.setGeometry(200, 320, 200, 30)

        self.diff_label = QtWidgets.QLabel("Difference:", self)
        self.diff_label.setGeometry(200, 390, 200, 30)

        self.spectrogram_path = None  # Initialize to None

    def generate_filename(self):
        # Generate a timestamped filename using current date and time
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return timestamp

    def recordCurrentAudio(self):
        fs = 44100  # Sample rate
        seconds = 5  # Duration of recording
        self.status_label.setText("Status: Recording...")
        self.repaint()

        # Record audio
        recording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
        sd.wait()  # Wait for the recording to finish
        
        # Generate unique filename for the audio based on timestamp
        timestamp = self.generate_filename()
        self.current_audio_path = os.path.join(SOUNDS_FOLDER, f"current_audio_{timestamp}.wav")
        write(self.current_audio_path, fs, recording)  # Save as WAV file
        self.status_label.setText("Status: Recording Complete")
        time.sleep(2)

        # Save to database
        #self.saveToDatabase()

    def playCurrentAudio(self):
        if not hasattr(self, "current_audio_path") or not os.path.exists(self.current_audio_path):
            self.status_label.setText("Status: No audio file to play!")
            return

        try:
            # Playback the recorded audio
            filename = self.current_audio_path
            self.status_label.setText("Status: Playing Audio")
            data, fs = sf.read(filename, dtype="float32")
            sd.play(data, fs)
            status = sd.wait()  # Wait until the sound finishes
            self.status_label.setText("Status: Playback Complete")
        except Exception as e:
            self.status_label.setText(f"Error: {str(e)}")

    def compareAudio(self):
        try:
            # Load the original sound for comparison
            original_path = os.path.join(BASE_DIR, "original_sounds", "original_audio.wav")
            if not os.path.exists(original_path):
                self.status_label.setText("Status: Original audio not found!")
                return

            # Load original and current audio files
            x, sr1 = librosa.load(original_path, sr=None)
            y, sr2 = librosa.load(self.current_audio_path, sr=None)

            # Compute the Root Mean Square Error (RMSE) difference
            p = np.split(x, 10)
            q = np.split(y, 10)
            i = []
            aud1 = []
            aud2 = []
            for i in p:
                r = np.sqrt(np.mean(i**2))
                aud1.append(r)
            for i in q:
                s = np.sqrt(np.mean(i**2))
                aud2.append(s)
            x1 = np.asarray(aud1)
            y1 = np.asarray(aud2)
            j = 0
            k = []
            for j in range(len(x1)):
                d = (np.abs(x1[j] - y1[j])) * 100
                e = float('%.3f' % d)
                k.append(e)

            val = ' '.join([str(elem) for elem in k])
            self.values = val
            self.res = max(k)
            self.diff_label.setText("Difference: " + str(self.res) + "%")
            
            # Plot the comparison diagrams
            self.plotComparison(x, y, sr1, sr2)
            
        except Exception as e:
            self.status_label.setText(f"Error: {str(e)}")

    def plotComparison(self, x, y, sr1, sr2):
        plt.figure(figsize=(10, 8))

        # Plot waveform comparison
        plt.subplot(3, 1, 1)
        plt.plot(x, label="Original Audio", alpha=0.6)
        plt.plot(y, label="Compared Audio", alpha=0.6)
        plt.title("Waveform Comparison")
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.legend()

        # Plot spectrogram comparison
        plt.subplot(3, 1, 2)
        S1 = librosa.feature.melspectrogram(y=x, sr=sr1)
        S1_dB = librosa.power_to_db(S1, ref=np.max)
        librosa.display.specshow(S1_dB, sr=sr1, x_axis="time", y_axis="mel")
        plt.colorbar(format="%+2.0f dB")
        plt.title("Original Audio Spectrogram")

        plt.subplot(3, 1, 3)
        S2 = librosa.feature.melspectrogram(y=y, sr=sr2)
        S2_dB = librosa.power_to_db(S2, ref=np.max)
        librosa.display.specshow(S2_dB, sr=sr2, x_axis="time", y_axis="mel")
        plt.colorbar(format="%+2.0f dB")
        plt.title("Compared Audio Spectrogram")

        plt.tight_layout()

        # Save the waveform image
        timestamp = self.generate_filename()
        waveform_path = os.path.join(WAVEFORMS_FOLDER, f"waveform_{timestamp}.png")
        plt.savefig(waveform_path)  # Save the waveform image

        # Save all paths to the database
        self.spectrogram_path = os.path.join(SPECTROGRAMS_FOLDER, f"spectrogram_{timestamp}.png")
        plt.savefig(self.spectrogram_path)  # Save the spectrogram image
        self.current_audio_path = os.path.join(SOUNDS_FOLDER, f"current_audio_{timestamp}.wav")

        # Save to the database
        self.saveComparisonToDatabase(waveform_path)

        plt.show()

    def showSpectrogram(self):
        if not hasattr(self, "current_audio_path") or not os.path.exists(self.current_audio_path):
            self.status_label.setText("Status: No audio to display!")
            return

        try:
            y, sr = librosa.load(self.current_audio_path, sr=None)
            S = librosa.feature.melspectrogram(y=y, sr=sr)
            S_dB = librosa.power_to_db(S, ref=np.max)
            plt.figure(figsize=(10, 4))
            librosa.display.specshow(S_dB, sr=sr, x_axis="time", y_axis="mel")
            plt.colorbar(format="%+2.0f dB")
            plt.title("Mel-frequency spectrogram")
            
            # Generate a timestamped filename for the spectrogram
            timestamp = self.generate_filename()
            self.spectrogram_path = os.path.join(SPECTROGRAMS_FOLDER, f"spectrogram_{timestamp}.png")
            plt.savefig(self.spectrogram_path)  # Save the spectrogram image

            # Save the spectrogram path to the database
            #self.saveToDatabase()

            plt.show()
        except Exception as e:
            self.status_label.setText(f"Error: {str(e)}")

    def saveComparisonToDatabase(self, waveform_path):
        # Create a session to interact with the database
        db = SessionLocal()
        try:
            # Find the original sound by id (assuming id=1)
            original_sound = db.query(OriginalSound).filter(OriginalSound.id == 1).first()
            if not original_sound:
                self.status_label.setText("Original sound not found!")
                return

            # Save the comparison sound details, including the difference percentage
            new_comparison = ComparisonSound(
                original_sound_id=original_sound.id,
                file_path=self.current_audio_path,
                spectrogram_path=self.spectrogram_path,
                waveform_path=waveform_path,
                difference_percentage=self.res,  # Save the difference percentage
                recorded_at=func.now()  # Store the current timestamp
            )
            db.add(new_comparison)
            db.commit()
            self.status_label.setText(f"Status: Comparison Saved (ID: {new_comparison.id})")
        except Exception as e:
            self.status_label.setText(f"Error saving comparison to DB: {str(e)}")
            db.rollback()
        finally:
            db.close()


if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    window = App2CompareSound()
    window.show()
    app.exec_()
