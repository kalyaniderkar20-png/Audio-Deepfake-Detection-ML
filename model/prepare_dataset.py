import os
import pandas as pd
import librosa

from audio_processing.feature_extraction import extract_features


# Dataset folders
REAL_FOLDER = "dataset/real"
FAKE_FOLDER = "dataset/fake"


# Supported audio extensions
SUPPORTED_EXTENSIONS = (
    ".wav",
    ".flac",
    ".mp3",
    ".m4a"
)


dataset = []


def process_folder(folder_path, label):

    print("Searching:", folder_path)

    for root, dirs, files in os.walk(folder_path):

        for filename in files:

            if filename.lower().endswith(SUPPORTED_EXTENSIONS):

                file_path = os.path.join(root, filename)

                print("Processing:", file_path)

                try:

                    audio, sample_rate = librosa.load(
                        file_path,
                        sr=None
                    )


                    features = extract_features(
                        audio,
                        sample_rate
                    )


                    if features is None:
                        print("❌ Feature extraction failed:", filename)
                        continue


                    row = features.tolist()

                    row.append(label)
                    row.append(filename)

                    dataset.append(row)

                    print("✓ Success:", filename)


                except Exception as e:

                    print("✗ Error:", filename)
                    print(e)



print("Reading REAL audio files...")
process_folder(REAL_FOLDER, 0)


print("Reading FAKE audio files...")
process_folder(FAKE_FOLDER, 1)



if len(dataset) == 0:

    print("\n❌ No audio files found!")
    print("Check dataset/real and dataset/fake folders")

else:

    columns = [
        f"Feature_{i}" 
        for i in range(len(dataset[0])-2)
    ]

    columns += [
        "Label",
        "File"
    ]


    df = pd.DataFrame(
        dataset,
        columns=columns
    )


    os.makedirs("dataset", exist_ok=True)

    df.to_csv(
        "dataset/features.csv",
        index=False
    )


    print("\n================================")
    print("Dataset created successfully!")
    print("================================")

    print(df.head())

    print("Total samples:", len(df))