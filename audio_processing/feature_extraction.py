import librosa
import numpy as np
from scipy.stats import skew, kurtosis
from audio_processing.preprocessing import preprocess_audio


def get_feature_names():
    """
    Returns the exact list of 96 canonical feature names in strict order.
    Used identically across dataset creation, training, testing, and Streamlit inference.
    """
    names = []
    # 1-40: MFCC Means (40)
    names.extend([f"MFCC_{i+1}" for i in range(40)])
    # 41-52: Delta MFCC Means (12 - First Derivative: Pitch Velocity)
    names.extend([f"Delta_MFCC_{i+1}" for i in range(12)])
    # 53-64: Delta-Delta MFCC Means (12 - Second Derivative: Pitch Acceleration)
    names.extend([f"Delta2_MFCC_{i+1}" for i in range(12)])
    # 65-76: Chroma STFT Means (12 - Pitch Class Distribution)
    names.extend([f"Chroma_{i+1}" for i in range(12)])
    # 77-83: Spectral Contrast Means (7 - Frequency Sub-band Peaks vs Valleys)
    names.extend([f"Spectral_Contrast_{i+1}" for i in range(7)])
    # 84-96: Physical, Harmonics & Vocoder Phase Artifact Descriptors (13)
    names.extend([
        "Spectral_Centroid",     # Frequency Brightness / Center of Mass
        "Spectral_Bandwidth",    # Spectral Spread / Bandwidth
        "Spectral_Rolloff",      # 85% Power Roll-off Frequency
        "Zero_Crossing_Rate",    # Frame Zero Crossing Density
        "RMS_Energy",            # Signal Root Mean Square Power
        "Spectral_Flatness",     # Noise-like vs Tone-like Measure
        "Spectral_Slope",        # High-Frequency Spectral Tilt
        "HF_Energy_Ratio",       # Vocoder Cutoff High Frequency Energy Ratio (>6.5kHz)
        "ZCR_Std_Dev",          # Zero Crossing Rate Variance (Human Jitter vs AI Uniformity)
        "Pitch_Jitter_Std",    # Pitch Autocorrelation Jitter Variance
        "Amplitude_Shimmer",   # Peak-to-Peak Amplitude Shimmer Variance
        "Spectral_Skewness",   # Spectral Envelope Skewness
        "Spectral_Kurtosis"    # Spectral Envelope Peakiness (Kurtosis)
    ])
    return names


def extract_features(audio, sr=16000, return_dict=False, do_preprocess=True):
    """
    Extracts 96 canonical acoustic features from audio waveform.
    Matches get_feature_names() 100% in count, order, and naming.
    """
    try:
        if audio is None or len(audio) == 0:
            return None

        if do_preprocess:
            audio = preprocess_audio(audio, sr=sr)

        if audio is None or len(audio) == 0:
            return None

        # 1. MFCCs (40 Means)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        mfcc_mean = np.mean(mfcc, axis=1)

        # 2. Delta MFCCs (12 Means)
        delta_mfcc = librosa.feature.delta(mfcc[:12])
        delta_mfcc_mean = np.mean(delta_mfcc, axis=1)

        # 3. Delta2 MFCCs (12 Means)
        delta2_mfcc = librosa.feature.delta(mfcc[:12], order=2)
        delta2_mfcc_mean = np.mean(delta2_mfcc, axis=1)

        # 4. Chroma STFT (12 Means)
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)

        # 5. Spectral Contrast (7 Means)
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

        # 9. Zero Crossing Rate Mean & Std
        zcr = librosa.feature.zero_crossing_rate(y=audio)
        zcr_mean = np.array([np.mean(zcr)])
        zcr_std = np.array([np.std(zcr)])

        # 10. RMS Energy
        rms = librosa.feature.rms(y=audio)
        rms_mean = np.array([np.mean(rms)])

        # 11. Spectral Flatness
        flatness = librosa.feature.spectral_flatness(y=audio)
        flatness_mean = np.array([np.mean(flatness)])

        # 12. Spectral Slope
        S = np.abs(librosa.stft(audio))
        freqs = librosa.fft_frequencies(sr=sr)
        S_mean = np.mean(S, axis=1)
        slope = np.polyfit(freqs, S_mean, 1)[0] if len(freqs) == len(S_mean) else 0.0
        slope_mean = np.array([slope])

        # 13. High-Frequency Energy Ratio (>6.5 kHz)
        hf_bin = int((6500 / (sr / 2)) * len(S_mean))
        hf_energy = np.sum(S_mean[hf_bin:]) if hf_bin < len(S_mean) else 0.0
        total_energy = np.sum(S_mean) + 1e-9
        hf_ratio = np.array([hf_energy / total_energy])

        # 14. Pitch Jitter Estimate
        autocorr = np.correlate(audio, audio, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        jitter_estimate = np.array([np.std(autocorr[:int(sr*0.05)]) if len(autocorr) > int(sr*0.05) else 0.0])

        # 15. Amplitude Shimmer Estimate
        frame_len = int(sr * 0.02)
        if len(audio) >= frame_len:
            frames = librosa.util.frame(audio, frame_length=frame_len, hop_length=frame_len//2)
            frame_peaks = np.max(np.abs(frames), axis=0)
            shimmer_estimate = np.array([np.std(frame_peaks)])
        else:
            shimmer_estimate = np.array([0.0])

        # 16. Spectral Skewness & Kurtosis
        spec_skew = np.array([float(skew(S_mean))])
        spec_kurt = np.array([float(kurtosis(S_mean))])

        # Combine all 96 features in exact canonical order
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
            slope_mean,
            hf_ratio,
            zcr_std,
            jitter_estimate,
            shimmer_estimate,
            spec_skew,
            spec_kurt
        ])

        if return_dict:
            feature_names = get_feature_names()
            return dict(zip(feature_names, features))

        return features.astype(np.float32)

    except Exception as e:
        print(f"❌ Feature Extraction Error: {e}")
        return None
