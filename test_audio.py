import os
import sys

# Ensure current project directory is prioritized in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import joblib
import pandas as pd
import numpy as np
import librosa
from audio_processing.preprocessing import preprocess_audio
from audio_processing.feature_extraction import extract_features, get_feature_names

# Fix Windows terminal UTF-8 encoding
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


def test_audio_file(file_path):
    if not os.path.exists(file_path):
        print(f"❌ Error: File '{file_path}' does not exist.")
        return

    model_path = "model/audio_deepfake_model.pkl"
    scaler_path = "model/scaler.pkl"
    feature_names_path = "model/feature_names.pkl"

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        print("❌ Error: Trained model or scaler artifact missing in model/")
        return

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    saved_feature_names = joblib.load(feature_names_path) if os.path.exists(feature_names_path) else get_feature_names()

    print("============================================================")
    print("       AI VOICESHIELD - FORENSIC PREDICTION TEST            ")
    print("============================================================")

    # 1. Load Audio
    try:
        audio, sr = librosa.load(file_path, sr=16000, mono=True)
        duration = len(audio) / sr
        print(f"✅ Audio loaded: {os.path.basename(file_path)} ({duration:.2f}s, {sr}Hz)")
    except Exception as e:
        print(f"❌ Audio Loading Error: {e}")
        return

    # 2. Preprocess Signal
    clean_audio = preprocess_audio(audio, sr=sr)

    # 3. Extract Features
    raw_feats = extract_features(clean_audio, sr=sr, do_preprocess=False)
    if raw_feats is None:
        print("❌ Feature extraction failed.")
        return

    # 4. Construct Feature DataFrame matching saved feature_names 100%
    expected_n = len(saved_feature_names)
    vec = raw_feats[:expected_n] if len(raw_feats) >= expected_n else np.pad(raw_feats, (0, expected_n - len(raw_feats)))
    features_df = pd.DataFrame([vec], columns=saved_feature_names)

    # 5. Feature Validation Check
    if list(features_df.columns) != list(saved_feature_names):
        print("⚠️ Feature mismatch detected! Re-aligning feature columns...")
        features_df = features_df[saved_feature_names]

    # 6. Scale Features & Predict
    scaled_input = scaler.transform(features_df)
    probabilities = model.predict_proba(scaled_input)[0]

    real_prob = probabilities[0] * 100
    fake_prob = probabilities[1] * 100

    # Pure Binary Classification Thresholding (Real vs Deepfake)
    if fake_prob >= 50.0:
        verdict = "🔴 AI GENERATED DEEPFAKE DETECTED"
        risk_level = "HIGH RISK (DEEPFAKE)"
        confidence = fake_prob
    else:
        verdict = "🟢 REAL HUMAN VOICE (AUTHENTIC)"
        risk_level = "LOW RISK (AUTHENTIC REAL)"
        confidence = real_prob

    print("\n============================================================")
    print("                FORENSIC PREDICTION RESULT                  ")
    print("============================================================")
    print(f"  🟢 Real Voice Likelihood : {real_prob:6.2f}%")
    print(f"  🔴 AI Deepfake Likelihood: {fake_prob:6.2f}%")
    print(f"  📊 Confidence Score      : {confidence:6.2f}%")
    print(f"  ⚠️ Risk Status           : {risk_level}")
    print("------------------------------------------------------------")
    print(f"  PERDICT: {perdict}")
    print("============================================================\n")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_audio_file(sys.argv[1])
    else:
        # Default test on Kalyani, Rutuja, AI Hindi, and fakevoice
        test_files = [
            "uploads/kalyani_marathi_voice.ogg",
            "uploads/rutuja_dirkar_real_voice.ogg",
            "uploads/ai_hindi_friendship_day.ogg",
            "uploads/fakevoice.mpeg",
            "uploads/file_example_WAV_5MG.wav"
        ]
        for f in test_files:
            if os.path.exists(f):
                test_audio_file(f)
