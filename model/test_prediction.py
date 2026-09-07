import librosa
import numpy as np
import joblib

from audio_processing.feature_extraction import extract_features


MODEL_PATH = "model/audio_deepfake_model.pkl"

# Load trained model
model = joblib.load(MODEL_PATH)

print("=" * 50)
print("       AI VOICESHIELD - PREDICTION TEST")
print("=" * 50)

file_path = input("\nEnter path of WAV audio: ").strip().strip('"')

try:

    # Same loading method used during training
    audio, sample_rate = librosa.load(
        file_path,
        sr=None
    )

    print("\nAudio loaded successfully!")
    print("Sample rate:", sample_rate)
    print("Duration:", round(len(audio) / sample_rate, 2), "seconds")

    # Extract same 64 features
    features = extract_features(
        audio,
        sample_rate
    )

    if features is None:
        print("❌ Feature extraction failed.")
        exit()

    print("Feature shape:", features.shape)
    print("Feature count:", len(features))

    # Prepare model input
    X = np.array(features).reshape(1, -1)

    print("Model input shape:", X.shape)

    # Prediction
    prediction = model.predict(X)[0]

    # Prediction probabilities
    probabilities = model.predict_proba(X)[0]

    print("\n" + "=" * 50)
    print("          PREDICTION RESULT")
    print("=" * 50)

    print("Prediction:", prediction)

    print("\nProbabilities:")

    for class_name, probability in zip(
        model.classes_,
        probabilities
    ):

        if class_name == 0:
            print(f"REAL: {probability:.2%}")
        else:
            print(f"FAKE: {probability:.2%}")

    if prediction == 0:
        print("\n🔵 RESULT: REAL AUDIO")
    else:
        print("\n🔴 RESULT: AI-GENERATED / FAKE AUDIO")

except Exception as e:

    print("\n❌ Prediction error:")
    print(e)