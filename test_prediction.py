import os
import sys
import pandas as pd
import librosa
import numpy as np
import joblib
from audio_processing.preprocessing import preprocess_audio
from audio_processing.feature_extraction import extract_features, get_feature_names

# Fix Windows terminal UTF-8 encoding
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

MODEL_PATH = "model/audio_deepfake_model.pkl"
SCALER_PATH = "model/scaler.pkl"

if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    print("❌ Model or Scaler missing. Please run 'python train_model.py' first.")
    sys.exit(1)

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

print("=" * 60)
print("       AI VOICESHIELD - FORENSIC PREDICTION TEST")
print("=" * 60)

if len(sys.argv) > 1:
    file_path = sys.argv[1]
else:
    file_path = input("\nEnter path of audio file (.wav/.mp3/.flac/.m4a/.mpeg): ").strip().strip('"')

if not os.path.exists(file_path):
    print(f"❌ File not found: '{file_path}'")
    sys.exit(1)

try:
    audio, sample_rate = librosa.load(file_path, sr=16000, mono=True)

    print("\n✅ Audio loaded successfully!")
    print(f"   Sample Rate : {sample_rate} Hz")
    print(f"   Raw Duration: {len(audio) / sample_rate:.2f} seconds")

    clean_audio = preprocess_audio(audio, sr=16000)
    features = extract_features(clean_audio, sr=16000, do_preprocess=False)

    if features is None:
        print("❌ Feature extraction failed.")
        sys.exit(1)

    feature_names = get_feature_names()
    expected_n = getattr(scaler, "n_features_in_", len(features))
    vec = features[:expected_n] if len(features) >= expected_n else np.pad(features, (0, expected_n - len(features)))
    names = feature_names[:expected_n] if len(feature_names) >= expected_n else [f"Feature_{i+1}" for i in range(expected_n)]

    X_df = pd.DataFrame([vec], columns=names)

    X_scaled = scaler.transform(X_df)

    prediction = model.predict(X_scaled)[0]
    probabilities = model.predict_proba(X_scaled)[0]

    real_prob = probabilities[0] * 100
    fake_prob = probabilities[1] * 100

    print("\n" + "=" * 60)
    print("          FORENSIC PREDICTION RESULT")
    print("=" * 60)

    print(f"  🟢 Real Voice Likelihood : {real_prob:.2f}%")
    print(f"  🔴 AI Deepfake Likelihood: {fake_prob:.2f}%")

    if prediction == 0:
        print("\n🔵 VERDICT: REAL HUMAN VOICE (AUTHENTIC)")
    else:
        print("\n🔴 VERDICT: AI-GENERATED / DEEPFAKE VOICE")

    print("=" * 60)

except Exception as e:
    print(f"\n❌ Prediction Error: {e}")