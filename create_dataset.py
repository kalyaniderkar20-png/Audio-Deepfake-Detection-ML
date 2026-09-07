import os
import sys
import pandas as pd
import librosa
from audio_processing.preprocessing import preprocess_audio, augment_audio
from audio_processing.feature_extraction import extract_features, get_feature_names

# Fix Windows terminal UTF-8 encoding
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Dataset folders
REAL_FOLDER = "dataset/real"
FAKE_FOLDER = "dataset/fake"

# Supported audio extensions
SUPPORTED_EXTENSIONS = (
    ".wav",
    ".flac",
    ".mp3",
    ".m4a",
    ".mpeg",
    ".ogg"
)


def create_feature_csv():
    dataset = []

    def process_folder(folder_path, label):
        print(f"[SEARCH] Searching: {folder_path}")

        if not os.path.exists(folder_path):
            print(f"[WARN] Directory '{folder_path}' does not exist.")
            return

        for root, dirs, files in os.walk(folder_path):
            for filename in files:
                if filename.lower().endswith(SUPPORTED_EXTENSIONS):
                    file_path = os.path.join(root, filename)

                    try:
                        # 1. Load audio standardized at 16000 Hz
                        audio, sample_rate = librosa.load(file_path, sr=16000)

                        # 2. Preprocess audio (silence trimming & pre-emphasis filtering)
                        clean_audio = preprocess_audio(audio, sr=16000)

                        if clean_audio is None or len(clean_audio) == 0:
                            continue

                        # 3. Data Augmentation (original + noise + pitch shift)
                        augmented_signals = augment_audio(clean_audio, sr=16000)

                        for aug_idx, aug_sig in enumerate(augmented_signals):
                            features = extract_features(aug_sig, sr=16000, do_preprocess=False)

                            if features is None:
                                continue

                            row = features.tolist()
                            row.append(label)
                            aug_filename = f"{filename}_aug{aug_idx}" if aug_idx > 0 else filename
                            row.append(aug_filename)

                            dataset.append(row)

                        print(f"  [OK] Processed & Augmented ({len(augmented_signals)}x): {filename}")

                    except Exception as e:
                        print(f"  [ERROR] {filename}: {e}")

    print("[REAL] Reading & Preprocessing REAL audio files...")
    process_folder(REAL_FOLDER, 0)

    print("[FAKE] Reading & Preprocessing FAKE audio files...")
    process_folder(FAKE_FOLDER, 1)

    if len(dataset) == 0:
        print("\n[ERROR] No audio files found! Please ensure audio files exist in dataset/real and dataset/fake.")
        return False

    feature_names = get_feature_names()
    columns = feature_names + ["Label", "File"]

    df = pd.DataFrame(dataset, columns=columns)
    os.makedirs("dataset", exist_ok=True)
    csv_path = "dataset/features.csv"
    df.to_csv(csv_path, index=False)

    print("\n==========================================")
    print(f"[SUCCESS] Preprocessed Feature Dataset Created: {csv_path}")
    print("==========================================")
    print(df.head())
    print(f"[SUMMARY] Total Dataset Samples (with Data Augmentation): {len(df)} (Features: {len(feature_names)})")
    return True


if __name__ == "__main__":
    create_feature_csv()
