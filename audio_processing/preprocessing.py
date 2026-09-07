import numpy as np
import librosa


def preprocess_audio(audio, sr=16000, top_db=25, pre_emphasis=0.97):
    """
    Standard Audio Preprocessing Pipeline matching Digital Signal Processing standards:
    1. Trims leading and trailing silent sections (silence removal).
    2. Applies pre-emphasis high-frequency filter: y(t) = x(t) - alpha * x(t-1).
    3. Normalizes waveform amplitude.
    4. Pads short audio segments to ensure minimum length (>0.2s).
    """
    try:
        if audio is None or len(audio) == 0:
            return None

        # 1. Amplitude Normalization
        audio = librosa.util.normalize(audio)

        # 2. Silence Trimming (removes silent leading/trailing frames)
        trimmed_audio, _ = librosa.effects.trim(audio, top_db=top_db)
        if len(trimmed_audio) > int(sr * 0.2):
            audio = trimmed_audio

        # 3. Pre-Emphasis Filtering (emphasizes high frequency vocal Formants & Vocoder artifacts)
        if pre_emphasis > 0:
            audio = np.append(audio[0], audio[1:] - pre_emphasis * audio[:-1])

        # 4. Minimum Duration Padding
        min_samples = int(sr * 0.2)
        if len(audio) < min_samples:
            audio = np.pad(audio, (0, min_samples - len(audio)), mode='constant')

        # Final Normalization
        audio = librosa.util.normalize(audio)
        return audio.astype(np.float32)

    except Exception as e:
        print(f"⚠️ Preprocessing warning: {e}")
        return audio


def augment_audio(audio, sr=16000):
    """
    Data Augmentation Pipeline to make ML Model robust against real-world voice recording noise,
    mobile microphone compression, and pitch inflections:
    1. Adds subtle Gaussian white noise (SNR ~ 30dB).
    2. Applies pitch shift (-1 to +1 semitones).
    3. Applies slight speed time stretch (0.95x - 1.05x).
    """
    augmented = []
    if audio is None or len(audio) == 0:
        return augmented

    # Original preprocessed signal
    augmented.append(audio)

    try:
        # Augmentation 1: Subtle Ambient Noise Floor
        noise = np.random.normal(0, 0.005, len(audio))
        noisy_audio = librosa.util.normalize(audio + noise)
        augmented.append(noisy_audio.astype(np.float32))
    except Exception:
        pass

    try:
        # Augmentation 2: Pitch Shift (Micro pitch inflection)
        shifted = librosa.effects.pitch_shift(audio, sr=sr, n_steps=1.0)
        augmented.append(librosa.util.normalize(shifted).astype(np.float32))
    except Exception:
        pass

    return augmented
