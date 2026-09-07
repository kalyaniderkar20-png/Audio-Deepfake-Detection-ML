import os
import sys
import numpy as np
import soundfile as sf
from scipy.signal import butter, lfilter

# Fix Windows terminal UTF-8 encoding
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


def apply_formant_filters(signal, sr=16000):
    """
    Applies human vocal tract formant resonances (F1=500Hz, F2=1500Hz, F3=2500Hz, F4=3500Hz)
    to model realistic human speech acoustics.
    """
    formants = [500, 1500, 2500, 3500]
    bandwidths = [50, 80, 100, 120]
    output = np.zeros_like(signal)
    
    for f0, bw in zip(formants, bandwidths):
        low = max(20, f0 - bw // 2)
        high = min(sr // 2 - 100, f0 + bw // 2)
        b, a = butter(2, [low / (sr / 2), high / (sr / 2)], btype='band')
        output += lfilter(b, a, signal) * 0.25
        
    return signal * 0.4 + output * 0.6


def generate_speech_signal(is_fake=False, duration=3.5, sr=16000, seed=None):
    """
    Generates realistic speech-like acoustic signals modeling natural human voice harmonics
    and vocal tract formants versus AI vocoder / neural TTS deepfake acoustic artifacts.
    """
    if seed is not None:
        np.random.seed(seed)
        
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    
    if not is_fake:
        # --- REAL HUMAN VOICE ACOUSTICS ---
        # 1. Natural fundamental frequency (F0) with pitch jitter, micro-vibrato, and contour inflection
        f0_base = 120 + 25 * np.sin(2 * np.pi * 0.8 * t) + 10 * np.cos(2 * np.pi * 2.2 * t)
        f0_jitter = 3.0 * np.random.normal(0, 1.0, len(t))
        f0 = f0_base + f0_jitter
        phase = 2 * np.pi * np.cumsum(f0) / sr
        
        # 2. Harmonic series with natural energy fall-off
        harmonics = np.zeros_like(t)
        for h in range(1, 12):
            amplitude = (1.0 / h**1.2) * (1.0 + 0.1 * np.sin(2 * np.pi * 3.0 * t + h))
            harmonics += amplitude * np.sin(h * phase)
            
        # 3. Apply vocal tract formant resonance filter
        speech_raw = apply_formant_filters(harmonics, sr=sr)
        
        # 4. Natural speech articulation envelope
        cadence = 0.5 + 0.5 * np.sin(2 * np.pi * 2.8 * t)**2
        pause_mask = (np.sin(2 * np.pi * 0.7 * t) > -0.4).astype(float)
        envelope = cadence * pause_mask
        
        # 5. Natural ambient noise floor & room acoustics
        ambient = np.random.normal(0, 0.005, len(t))
        signal = speech_raw * envelope + ambient
        
    else:
        # --- AI DEEPFAKE / NEURAL VOCODER ACOUSTICS ---
        # 1. Unnaturally rigid / monotone F0 pitch trajectory
        f0_base = 130 + 1.2 * np.sin(2 * np.pi * 0.4 * t)
        phase = 2 * np.pi * np.cumsum(f0_base) / sr
        
        # 2. Vocoder phase distortion & synthetic harmonic overtones
        harmonics = np.zeros_like(t)
        for h in range(1, 10):
            phase_mismatch = (h * 0.08) if h > 3 else 0.0
            harmonics += (1.0 / h**0.95) * np.sin(h * phase + phase_mismatch)
            
        # 3. High-frequency vocoder spectral noise cutoff & metallic overtone
        hf_vocoder_spike = 0.09 * np.sin(2 * np.pi * 6700 * t) + 0.06 * np.sin(2 * np.pi * 7200 * t)
        hf_noise_floor = np.random.normal(0, 0.015, len(t))
        
        # 4. Flat artificial envelope
        envelope = 0.75 + 0.15 * np.sin(2 * np.pi * 3.2 * t)
        signal = (harmonics + hf_vocoder_spike) * envelope + hf_noise_floor

    # Amplitude normalization
    max_val = np.max(np.abs(signal))
    if max_val > 0:
        signal = signal / max_val * 0.9

    return signal.astype(np.float32), sr


def build_synthetic_dataset(num_real=100, num_fake=100):
    real_dir = "dataset/real"
    fake_dir = "dataset/fake"
    samples_dir = "samples"

    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(fake_dir, exist_ok=True)
    os.makedirs(samples_dir, exist_ok=True)

    print(f"[REAL] Generating {num_real} Real Human Voice Audio Samples...")
    for i in range(num_real):
        audio, sr = generate_speech_signal(is_fake=False, duration=3.5, seed=1000 + i)
        file_path = os.path.join(real_dir, f"real_voice_{i+1:03d}.wav")
        sf.write(file_path, audio, sr)

    print(f"[FAKE] Generating {num_fake} Deepfake AI Audio Samples...")
    for i in range(num_fake):
        audio, sr = generate_speech_signal(is_fake=True, duration=3.5, seed=5000 + i)
        file_path = os.path.join(fake_dir, f"fake_voice_{i+1:03d}.wav")
        sf.write(file_path, audio, sr)

    print("[SAMPLES] Generating High-Fidelity Demo Samples for Streamlit UI...")
    real_demo, sr = generate_speech_signal(is_fake=False, duration=3.5, seed=9999)
    fake_demo, sr = generate_speech_signal(is_fake=True, duration=3.5, seed=8888)

    sf.write(os.path.join(samples_dir, "human_voice_sample.wav"), real_demo, sr)
    sf.write(os.path.join(samples_dir, "ai_deepfake_sample.wav"), fake_demo, sr)

    print("[SUCCESS] Realistic Speech Dataset and Sample Generation Complete!")


if __name__ == "__main__":
    build_synthetic_dataset(num_real=100, num_fake=100)
