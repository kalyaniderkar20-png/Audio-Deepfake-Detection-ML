import numpy as np
import librosa
from scipy.signal import butter, lfilter


def preprocess_audio(audio, sr=16000, top_db=25, pre_emphasis=0.97):
    """
    Standardized Audio Preprocessing Pipeline:
    1. Amplitude normalization.
    2. Silence trimming (top_db threshold).
    3. Pre-emphasis highpass filtering: y(t) = x(t) - 0.97 * x(t-1).
    4. Minimum duration padding (>0.25s).
    """
    try:
        if audio is None or len(audio) == 0:
            return None

        # 1. Amplitude Normalization
        audio = librosa.util.normalize(audio)

        # 2. Silence Trimming
        trimmed_audio, _ = librosa.effects.trim(audio, top_db=top_db)
        if len(trimmed_audio) >= int(sr * 0.25):
            audio = trimmed_audio

        # 3. Pre-Emphasis Highpass Filter
        if pre_emphasis > 0 and len(audio) > 1:
            audio = np.append(audio[0], audio[1:] - pre_emphasis * audio[:-1])

        # 4. Minimum Duration Padding (at least 0.25s)
        min_samples = int(sr * 0.25)
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
    Realistic Audio Augmentation (Used ONLY on Training Set to prevent leakage):
    1. Clean original signal.
    2. Subtle ambient Gaussian noise floor (SNR ~30dB).
    3. Mobile telephone/mic bandpass filter (300Hz - 7000Hz).
    4. Micro pitch shift (+1.0 semitones).
    """
    augmented = []
    if audio is None or len(audio) == 0:
        return augmented

    # Original signal
    augmented.append(audio)

    try:
        # Augmentation 1: Subtle Noise Floor
        noise = np.random.normal(0, 0.005, len(audio))
        noisy = librosa.util.normalize(audio + noise)
        augmented.append(noisy.astype(np.float32))
    except Exception:
        pass

    try:
        # Augmentation 2: Mobile Mic Bandpass Filter (300Hz - 7000Hz)
        lowcut, highcut = 300, 7000
        nyq = 0.5 * sr
        b, a = butter(2, [lowcut / nyq, highcut / nyq], btype='band')
        filtered = lfilter(b, a, audio)
        augmented.append(librosa.util.normalize(filtered).astype(np.float32))
    except Exception:
        pass

    try:
        # Augmentation 3: Micro Pitch Shift
        shifted = librosa.effects.pitch_shift(audio, sr=sr, n_steps=1.0)
        augmented.append(librosa.util.normalize(shifted).astype(np.float32))
    except Exception:
        pass

    return augmented
