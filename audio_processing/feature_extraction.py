import librosa
import numpy as np
from audio_processing.preprocessing import preprocess_audio


def get_feature_names():
    """
    Returns the exact list of feature names extracted by extract_features.
    Total: 91 acoustic signal features.
    """
    names = []
    # 40 MFCC Mean features
    names.extend([f"MFCC_{i+1}" for i in range(40)])
    # 12 Delta MFCC Mean features (first derivative: pitch velocity)
    names.extend([f"Delta_MFCC_{i+1}" for i in range(12)])
    # 12 Delta-Delta MFCC Mean features (second derivative: pitch acceleration)
    names.extend([f"Delta2_MFCC_{i+1}" for i in range(12)])
    # 12 Chroma STFT features (pitch classes)
    names.extend([f"Chroma_{i+1}" for i in range(12)])
    # 7 Spectral Contrast features (frequency sub-bands)
    names.extend([f"Spectral_Contrast_{i+1}" for i in range(7)])
    # Additional Spectral, Physical & Vocoder Artifact Descriptors
    names.extend([
        "Spectral_Centroid",
        "Spectral_Bandwidth",
        "Spectral_Rolloff",
        "Zero_Crossing_Rate",
        "RMS_Energy",
        "Spectral_Flatness",
        "Spectral_Slope"
    ])
    return names


def extract_features(audio, sr=16000, return_dict=False, do_preprocess=True):
    """
    Extracts 91 comprehensive acoustic features from preprocessed audio signal.
    """
    try:
        if audio is None or len(audio) == 0:
            print("❌ Feature extraction error: Empty audio signal.")
            return None

        # Preprocess audio (trim silence, pre-emphasis, amplitude normalize)
        if do_preprocess:
            audio = preprocess_audio(audio, sr=sr)

        if audio is None or len(audio) == 0:
            return None

        # 1. MFCC (40 Coefficients)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        mfcc_mean = np.mean(mfcc, axis=1)

        # 2. Delta MFCC (12 Coefficients - Pitch Velocity)
        delta_mfcc = librosa.feature.delta(mfcc[:12])
        delta_mfcc_mean = np.mean(delta_mfcc, axis=1)

        # 3. Delta-Delta MFCC (12 Coefficients - Pitch Acceleration)
        delta2_mfcc = librosa.feature.delta(mfcc[:12], order=2)
        delta2_mfcc_mean = np.mean(delta2_mfcc, axis=1)

        # 4. Chroma STFT (12 Pitch Classes)
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)

        # 5. Spectral Contrast (7 Frequency Bands)
        contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
        contrast_mean = np.mean(contrast, axis=1)

        # 6. Spectral Centroid
        centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
        centroid_mean = np.array([np.mean(centroid)])

        # 7. Spectral Bandwidth
        bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
        bandwidth_mean = np.array([np.mean(bandwidth)])

        # 8. Spectral Rolloff
        rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
        rolloff_mean = np.array([np.mean(rolloff)])

        # 9. Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(y=audio)
        zcr_mean = np.array([np.mean(zcr)])

        # 10. RMS Energy
        rms = librosa.feature.rms(y=audio)
        rms_mean = np.array([np.mean(rms)])

        # 11. Spectral Flatness (distinguishes tonal voice from neural vocoder noise artifacts)
        flatness = librosa.feature.spectral_flatness(y=audio)
        flatness_mean = np.array([np.mean(flatness)])

        # 12. Spectral Slope / Tilt
        S = np.abs(librosa.stft(audio))
        freqs = librosa.fft_frequencies(sr=sr)
        S_mean = np.mean(S, axis=1)
        slope = np.polyfit(freqs, S_mean, 1)[0] if len(freqs) == len(S_mean) else 0.0
        slope_mean = np.array([slope])

        # Concatenate all 91 features
        features = np.concatenate([
            mfcc_mean,
            delta_mfcc_mean,
            delta2_mfcc_mean,
            chroma_mean,
            contrast_mean,
            centroid_mean,
            bandwidth_mean,
            rolloff_mean,
            zcr_mean,
            rms_mean,
            flatness_mean,
            slope_mean
        ])

        if return_dict:
            feature_names = get_feature_names()
            return dict(zip(feature_names, features))

        return features

    except Exception as e:
        print(f"❌ Feature Extraction Error: {e}")
        return None
