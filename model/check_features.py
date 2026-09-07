import librosa
import numpy as np

from audio_processing.feature_extraction import extract_features

audio_path = input("Enter WAV/audio path: ").strip().strip('"')

audio, sr = librosa.load(
    audio_path,
    sr=16000,
    mono=True
)

print("=" * 60)
print("AI VOICESHIELD - FEATURE CHECK")
print("=" * 60)

print("Sample rate:", sr)
print("Audio shape:", audio.shape)
print("Audio duration:", len(audio) / sr)

features = extract_features(audio, sr=sr)

print("\nFeature type:", type(features))

if features is None:
    print("❌ Feature extraction returned None")
    exit()

features = np.asarray(features)

print("Feature shape:", features.shape)
print("Feature ndim:", features.ndim)
print("Feature count:", features.size)

print("\nFirst 10 features:")
print(features.flatten()[:10])

print("\nAny NaN:", np.isnan(features).any())
print("Any Inf:", np.isinf(features).any())

print("=" * 60)