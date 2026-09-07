import joblib

print("=" * 60)
print("AI VOICESHIELD - MODEL CHECK")
print("=" * 60)

model = joblib.load("model/audio_deepfake_model.pkl")
scaler = joblib.load("model/scaler.pkl")

print("\nMODEL")
print("Type:", type(model))
print("Expected features:", getattr(model, "n_features_in_", "Not available"))
print("Classes:", model.classes_)

print("\nSCALER")
print("Expected features:", scaler.n_features_in_)

if hasattr(scaler, "feature_names_in_"):
    print("\nFEATURE NAMES:")
    for i, name in enumerate(scaler.feature_names_in_):
        print(i + 1, name)

print("\n" + "=" * 60)