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


def apply_formant_filters(signal, sr=16000, f1=500, f2=1500, f3=2500, f4=3500):
    """
    Applies human vocal tract formant resonances (F1, F2, F3, F4)
    to model realistic human speech acoustics across varied pitch registers.
    """
    formants = [f1, f2, f3, f4]
    bandwidths = [50, 80, 100, 120]
    output = np.zeros_like(signal)
    
    for f0_f, bw in zip(formants, bandwidths):
        low = max(20, f0_f - bw // 2)
        high = min(sr // 2 - 100, f0_f + bw // 2)
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
        # Vary F0 base pitch randomly across human speech range (Male 80-160Hz, Female 160-260Hz)
        base_pitch = np.random.uniform(85, 240)
        f0_base = base_pitch + 25 * np.sin(2 * np.pi * 0.8 * t) + 10 * np.cos(2 * np.pi * 2.2 * t)
        f0_jitter = 3.5 * np.random.normal(0, 1.0, len(t))  # Natural human pitch jitter
        f0 = f0_base + f0_jitter
        phase = 2 * np.pi * np.cumsum(f0) / sr
        
        # Harmonic series with natural energy fall-off
        harmonics = np.zeros_like(t)
        for h in range(1, 12):
            amplitude = (1.0 / h**1.25) * (1.0 + 0.12 * np.sin(2 * np.pi * 3.0 * t + h))
            harmonics += amplitude * np.sin(h * phase)
            
        # Apply vocal tract formant resonance filter (varied per speaker)
        f1_val = np.random.uniform(450, 650)
        f2_val = np.random.uniform(1300, 1700)
        speech_raw = apply_formant_filters(harmonics, sr=sr, f1=f1_val, f2=f2_val)
        
        # Natural speech articulation envelope
        cadence = 0.5 + 0.5 * np.sin(2 * np.pi * 2.8 * t)**2
        pause_mask = (np.sin(2 * np.pi * 0.7 * t) > -0.4).astype(float)
        envelope = cadence * pause_mask
        
        # Natural ambient noise floor & room acoustics
        ambient = np.random.normal(0, 0.005, len(t))
        signal = speech_raw * envelope + ambient
        
    else:
        # --- AI DEEPFAKE / NEURAL VOCODER ACOUSTICS ---
        base_pitch = np.random.uniform(110, 200)
        f0_base = base_pitch + 1.2 * np.sin(2 * np.pi * 0.4 * t)  # Unnaturally rigid F0
        phase = 2 * np.pi * np.cumsum(f0_base) / sr
        
        # Vocoder phase distortion & synthetic harmonic overtones
        harmonics = np.zeros_like(t)
        for h in range(1, 10):
            phase_mismatch = (h * 0.08) if h > 3 else 0.0
            harmonics += (1.0 / h**0.95) * np.sin(h * phase + phase_mismatch)
            
        # High-frequency vocoder spectral noise cutoff & metallic overtone
        hf_vocoder_spike = 0.09 * np.sin(2 * np.pi * 6700 * t) + 0.06 * np.sin(2 * np.pi * 7200 * t)
        hf_noise_floor = np.random.normal(0, 0.015, len(t))
        
        # Flat artificial envelope
        envelope = 0.75 + 0.15 * np.sin(2 * np.pi * 3.2 * t)
        signal = (harmonics + hf_vocoder_spike) * envelope + hf_noise_floor

    # Amplitude normalization
    max_val = np.max(np.abs(signal))
    if max_val > 0:
        signal = signal / max_val * 0.9

    return signal.astype(np.float32), sr


def build_synthetic_dataset(num_real=200, num_fake=200):
    real_dir = "dataset/real/generated_real"
    fake_dir = "dataset/fake/generated_fake"
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
    build_synthetic_dataset(num_real=200, num_fake=200)
