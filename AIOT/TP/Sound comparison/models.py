
from sqlalchemy import Column, Integer, String, ForeignKey, DateTime, Float, func
from sqlalchemy.orm import relationship
from dataset import Base

# Define the model for original sound recordings
class OriginalSound(Base):
    __tablename__ = "original_sounds"

    id = Column(Integer, primary_key=True, index=True)
    bridge_name = Column(String(255), nullable=False)
    recorded_at = Column(DateTime(timezone=True), server_default=func.now())
    file_path = Column(String(255), nullable=False)
    spectrogram_path = Column(String(255))  # Path to the spectrogram image

    # In the OriginalSound class
    comparisons = relationship("ComparisonSound", back_populates="original_sound")

class ComparisonSound(Base):
    __tablename__ = "comparison_sounds"

    id = Column(Integer, primary_key=True, index=True)
    original_sound_id = Column(Integer, ForeignKey("original_sounds.id"))
    file_path = Column(String(255), nullable=False)
    spectrogram_path = Column(String(255))
    waveform_path = Column(String)  # Path to the waveform image (new field)
    recorded_at = Column(DateTime(timezone=True), server_default=func.now())
    difference_percentage = Column(Float)  # New field for the difference percentage
    
    # Relationship to link the comparison sound with the original sound
    original_sound = relationship("OriginalSound", back_populates="comparisons")

    def __repr__(self):
        return f"<ComparisonSound(id={self.id}, original_sound_id={self.original_sound_id}, file_path={self.file_path}, spectrogram_path={self.spectrogram_path}, waveform_path={self.waveform_path}, recorded_at={self.recorded_at})>"
