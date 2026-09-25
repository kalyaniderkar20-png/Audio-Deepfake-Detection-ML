import os
import sys
import shutil
import pandas as pd
import librosa
from concurrent.futures import ThreadPoolExecutor
from audio_processing.preprocessing import preprocess_audio
from audio_processing.feature_extraction import extract_features, get_feature_names

# Fix Windows terminal UTF-8 encoding
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Ensure user custom voice files are copied to dataset subfolders
src_rutuja = r"C:\Users\kalya\.gemini\antigravity\brain\11d087f9-446c-4ad8-a553-6957ef474891\.user_uploaded\uploaded_media_1789233716572.ogg"
src_ai_hindi = r"C:\Users\kalya\.gemini\antigravity\brain\11d087f9-446c-4ad8-a553-6957ef474891\.user_uploaded\uploaded_media_1789233757954.ogg"
src_kalyani = r"C:\Users\kalya\.gemini\antigravity\brain\11d087f9-446c-4ad8-a553-6957ef474891\.user_uploaded\uploaded_media_1789229262459.ogg"

os.makedirs("uploads", exist_ok=True)
os.makedirs("dataset/real/verified_real", exist_ok=True)
os.makedirs("dataset/fake/verified_fake", exist_ok=True)

if os.path.exists(src_rutuja):
    shutil.copy2(src_rutuja, "uploads/rutuja_dirkar_real_voice.ogg")
    shutil.copy2(src_rutuja, "dataset/real/verified_real/rutuja_dirkar_real_voice.ogg")

if os.path.exists(src_kalyani):
    shutil.copy2(src_kalyani, "uploads/kalyani_marathi_voice.ogg")
    shutil.copy2(src_kalyani, "dataset/real/verified_real/kalyani_marathi_voice.ogg")

if os.path.exists(src_ai_hindi):
    shutil.copy2(src_ai_hindi, "uploads/ai_hindi_friendship_day.ogg")
    shutil.copy2(src_ai_hindi, "dataset/fake/verified_fake/ai_hindi_friendship_day.ogg")


def process_audio_file(entry):
    file_path, label, speaker_group = entry
    filename = os.path.basename(file_path)
    try:
        audio, sr = librosa.load(file_path, sr=16000, mono=True)
        clean_audio = preprocess_audio(audio, sr=16000)

        if clean_audio is None or len(clean_audio) == 0:
            return None

        features = extract_features(clean_audio, sr=16000, do_preprocess=False)
        if features is None:
            return None

        row = features.tolist()
        row.append(label)
        row.append(speaker_group)
        row.append(filename)
        return row

    except Exception as e:
        print(f"Error processing {filename}: {e}")
        return None


def get_direct_file_list():
    entries = []

    # Target REAL Audio Files Directly
    real_candidates = [
        "uploads/rutuja_dirkar_real_voice.ogg",
        "uploads/kalyani_marathi_voice.ogg",
        "uploads/my own voice.ogg",
        "uploads/myvoice.wav",
        "uploads/file_example_WAV_5MG.wav"
    ]
    if os.path.exists("dataset/real/verified_real"):
        for f in os.listdir("dataset/real/verified_real"):
            p = os.path.join("dataset/real/verified_real", f)
            if os.path.isfile(p) and p not in real_candidates:
                real_candidates.append(p)

    for p in real_candidates:
        if os.path.exists(p):
            fname = os.path.basename(p)
            group = f"real_{fname.split('.')[0]}"
            entries.append((p, 0, group))

    # Target FAKE Audio Files Directly
    fake_candidates = [
        "uploads/ai_hindi_friendship_day.ogg",
        "uploads/fakevoice.mpeg",
        "uploads/7021_5.wav"
    ]
    if os.path.exists("dataset/fake/verified_fake"):
        for f in os.listdir("dataset/fake/verified_fake"):
            p = os.path.join("dataset/fake/verified_fake", f)
            if os.path.isfile(p) and p not in fake_candidates:
                fake_candidates.append(p)

    for p in fake_candidates:
        if os.path.exists(p):
            fname = os.path.basename(p)
            group = f"fake_{fname.split('.')[0]}"
            entries.append((p, 1, group))

    return entries


def create_dataset():
    entries = get_direct_file_list()
    print(f"[FAST EXTRACT] Processing {len(entries)} target audio files in parallel...")

    dataset = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(process_audio_file, entries))
        for res in results:
            if res is not None:
                dataset.append(res)

    feature_names = get_feature_names()
    columns = feature_names + ["Label", "Group_ID", "File"]

    df = pd.DataFrame(dataset, columns=columns)
    os.makedirs("dataset", exist_ok=True)
    csv_path = "dataset/features.csv"
    df.to_csv(csv_path, index=False)

    print("\n==========================================")
    print(f"[SUCCESS] Fast 96-Feature Dataset Created: {csv_path}")
    print(f"[SUMMARY] Total Unique Audio Samples: {len(df)} | Features: {len(feature_names)}")
    print("==========================================\n")
    return True


if __name__ == "__main__":
    create_dataset()
