import os
import time
import sounddevice as sd
from scipy.io.wavfile import write
import librosa
import librosa.display
import matplotlib.pyplot as plt
import simpleaudio as sa
from PyQt5 import QtWidgets
import soundfile as sf
import numpy as np
from dataset import SessionLocal
from models import OriginalSound
from sqlalchemy import func

# Set up the base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Define the folder for storing audio and spectrogram
AUDIO_FOLDER = os.path.join(BASE_DIR, "original_sounds")

# Create the directory if it doesn't exist
if not os.path.exists(AUDIO_FOLDER):
    os.makedirs(AUDIO_FOLDER)


class App1OriginalSound(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Original Sound Recorder")
        self.setGeometry(100, 100, 600, 400)

        # GUI Elements
        self.record_button = QtWidgets.QPushButton("Record Original Sound", self)
        self.record_button.setGeometry(200, 50, 200, 50)
        self.record_button.clicked.connect(self.recordOriginalAudio)

        self.play_button = QtWidgets.QPushButton("Play Original Sound", self)
        self.play_button.setGeometry(200, 120, 200, 50)
        self.play_button.clicked.connect(self.playOriginalAudio)

        self.spectrogram_button = QtWidgets.QPushButton("Show Spectrogram", self)
        self.spectrogram_button.setGeometry(200, 190, 200, 50)
        self.spectrogram_button.clicked.connect(self.showSpectrogram)

        self.status_label = QtWidgets.QLabel("Status: Ready", self)
        self.status_label.setGeometry(200, 260, 200, 30)

    def recordOriginalAudio(self):
        fs = 44100  # Sample rate
        seconds = 5  # Duration of recording
        self.status_label.setText("Status: Recording...")
        self.repaint()

        # Record audio
        recording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
        sd.wait()  # Wait for the recording to finish
        self.audio_path = os.path.join(AUDIO_FOLDER, "original_audio.wav")
        write(self.audio_path, fs, recording)  # Save as WAV file
        self.status_label.setText("Status: Recording Complete")
        time.sleep(2)

        # Save to database
        #self.saveToDatabase()

    def playOriginalAudio(self):
        if not hasattr(self, "audio_path") or not os.path.exists(self.audio_path):
            self.status_label.setText("Status: No audio file to play!")
            return

        try:
            # Playback the recorded audio
            filename = self.audio_path
            self.status_label.setText("Status: Playing Audio")
            data, fs = sf.read(filename, dtype='float32')
            sd.play(data, fs)
            status = sd.wait()  # Wait until the sound finishes
            self.status_label.setText("Status: Playback Complete")
        except Exception as e:
            self.status_label.setText(f"Error: {str(e)}")

    def showSpectrogram(self):
        if not hasattr(self, "audio_path") or not os.path.exists(self.audio_path):
            self.status_label.setText("Status: No audio to display!")
            return

        try:
            y, sr = librosa.load(self.audio_path, sr=None)
            S = librosa.feature.melspectrogram(y=y, sr=sr)
            S_dB = librosa.power_to_db(S, ref=np.max)
            plt.figure(figsize=(10, 4))
            librosa.display.specshow(S_dB, sr=sr, x_axis="time", y_axis="mel")
            plt.colorbar(format="%+2.0f dB")
            plt.title("Mel-frequency spectrogram")
            
            # Save the spectrogram as a PNG image in the folder
            self.spectrogram_path = os.path.join(AUDIO_FOLDER, "original_spectrogram.png")
            plt.savefig(self.spectrogram_path)  # Save the spectrogram image

            # Save the spectrogram path to the database
            self.saveToDatabase()

            plt.show()
        except Exception as e:
            self.status_label.setText(f"Error: {str(e)}")

    def saveToDatabase(self):
        # Create a session to interact with the database
        db = SessionLocal()
        try:
            # Check if a record with id=1 exists
            existing_record = db.query(OriginalSound).filter(OriginalSound.id == 1).first()

            if existing_record:
                # Update the existing record
                existing_record.bridge_name = "Bridge 01"  # Update bridge name
                existing_record.file_path = self.audio_path  # Update audio file path
                existing_record.spectrogram_path = self.spectrogram_path  # Update spectrogram file path
                existing_record.recorded_at = func.now()  # Update the recorded_at timestamp
                db.commit()  # Commit the changes
                self.status_label.setText(f"Status: Updated Database (ID: 1)")

            else:
                # If no record with id=1, insert a new record
                new_record = OriginalSound(
                    bridge_name="Bridge 01",  # Placeholder for the bridge name
                    file_path=self.audio_path,
                    spectrogram_path=self.spectrogram_path,
                )
                db.add(new_record)
                db.commit()  # Save the new entry to the database
                self.status_label.setText(f"Status: Saved to Database (ID: {new_record.id})")

        except Exception as e:
            self.status_label.setText(f"Error saving to DB: {str(e)}")
            print(f"Error saving to DB: {str(e)}")
            db.rollback()

        finally:
            db.close()


if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    window = App1OriginalSound()
    window.show()
    app.exec_()
