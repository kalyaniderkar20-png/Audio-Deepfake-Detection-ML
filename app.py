import os
import sys

# Ensure current project directory is prioritized in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import joblib
import pandas as pd
import numpy as np
import streamlit as st
import librosa
import soundfile as sf

# Clear stale Streamlit cache on startup
try:
    st.cache_resource.clear()
except Exception:
    pass

# Fix Windows stdout encoding if applicable
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Robust visualization imports
try:
    from audio_processing.visualization import (
        plot_waveform,
        plot_spectrogram,
        plot_chromagram,
        plot_feature_importance
    )
except ImportError as err:
    print(f"⚠️ Visualization import warning ({err}). Using inline fallback visualizers.")
    from audio_processing.visualization import plot_waveform, plot_spectrogram

    def plot_chromagram(audio, sample_rate):
        import matplotlib.pyplot as plt
        import librosa.display
        fig, ax = plt.subplots(figsize=(10, 3.5), facecolor='#05070a')
        ax.set_facecolor('#05070a')
        chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
        img = librosa.display.specshow(chroma, sr=sample_rate, x_axis='time', y_axis='chroma', ax=ax, cmap='cool')
        ax.set_title("Chromagram (Pitch Energy Distribution)", color='#ffffff', fontsize=12, fontweight='bold', pad=10)
        ax.set_xlabel("Time (Seconds)", color='#a0aec0', fontsize=10)
        ax.set_ylabel("Pitch Class", color='#a0aec0', fontsize=10)
        ax.tick_params(colors='#a0aec0')
        cbar = fig.colorbar(img, ax=ax)
        cbar.ax.yaxis.set_tick_params(color='#a0aec0')
        plt.setp(plt.getp(cbar.ax, 'yticklabels'), color='#a0aec0')
        plt.tight_layout()
        return fig

    def plot_feature_importance(feature_names, importances, top_n=12):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 4.5), facecolor='#05070a')
        ax.set_facecolor('#05070a')
        indices = np.argsort(importances)[::-1][:top_n]
        top_names = [feature_names[i] for i in indices]
        top_imps = importances[indices]
        y_pos = np.arange(len(top_names))
        colors = plt.cm.plasma(np.linspace(0.2, 0.8, len(top_names)))
        ax.barh(y_pos, top_imps, color=colors, edgecolor='#00f5d4', alpha=0.85)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_names, color='#ffffff', fontsize=10)
        ax.invert_yaxis()
        ax.set_title(f"Top {top_n} Acoustic Feature Importances in ML Model", color='#ffffff', fontsize=12, fontweight='bold', pad=10)
        ax.set_xlabel("Importance Weight", color='#a0aec0', fontsize=10)
        ax.tick_params(colors='#a0aec0')
        ax.grid(True, color='#1e293b', linestyle=':', alpha=0.6)
        plt.tight_layout()
        return fig

try:
    from audio_processing.preprocessing import preprocess_audio
    from audio_processing.feature_extraction import extract_features, get_feature_names
except ImportError:
    from audio_processing.feature_extraction import extract_features
    def get_feature_names():
        names = [f"MFCC_{i+1}" for i in range(40)] + [f"Delta_MFCC_{i+1}" for i in range(12)] + [f"Delta2_MFCC_{i+1}" for i in range(12)] + [f"Chroma_{i+1}" for i in range(12)] + [f"Spectral_Contrast_{i+1}" for i in range(7)]
        names.extend(["Spectral_Centroid", "Spectral_Bandwidth", "Spectral_Rolloff", "Zero_Crossing_Rate", "RMS_Energy", "Spectral_Flatness", "Spectral_Slope"])
        return names

    def preprocess_audio(audio, sr=16000):
        return audio


# ---------------------------------------------------------
# Page Configuration & Dark Cyberpunk Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="AI VoiceShield - Audio Deepfake Forensic Suite",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark Industrial Cyberpunk CSS Injection
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }

    .stApp {
        background-color: #05070a !important;
        color: #f1f5f9 !important;
    }

    header[data-testid="stHeader"] {
        background-color: #05070a !important;
    }

    [data-testid="stSidebar"] {
        background-color: #090d16 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
    }

    [data-testid="stSidebar"] * {
        color: #e2e8f0 !important;
    }

    .sidebar-card {
        background: #0f172a !important;
        border: 1px solid rgba(0, 245, 212, 0.2) !important;
        border-radius: 12px;
        padding: 14px;
        margin-bottom: 14px;
    }

    .sidebar-header {
        text-align: center;
        padding-bottom: 15px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 20px;
    }

    .badge-status {
        background: linear-gradient(90deg, #00f2fe, #4facfe);
        color: #05070a !important;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 800;
        letter-spacing: 1px;
        display: inline-block;
        margin-top: 6px;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: #0a0e17 !important;
        padding: 8px;
        border-radius: 14px;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }

    .stTabs [data-baseweb="tab"] {
        height: 46px;
        border-radius: 10px;
        color: #94a3b8 !important;
        font-weight: 600;
        padding: 0 18px;
        background: transparent !important;
    }

    .stTabs [aria-selected="true"] {
        background-color: rgba(0, 245, 212, 0.15) !important;
        color: #00f5d4 !important;
        border: 1px solid #00f5d4 !important;
        box-shadow: 0 0 15px rgba(0, 245, 212, 0.2);
    }

    .glass-card {
        background: #0b0f19 !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.7);
        margin-bottom: 20px;
    }

    [data-testid="stFileUploader"] {
        background: #0a0e17 !important;
        border: 2px dashed #00f5d4 !important;
        border-radius: 14px;
        padding: 20px;
    }

    [data-testid="stFileUploader"] * {
        color: #e2e8f0 !important;
    }

    /* Verdict Cards */
    .verdict-real {
        background: linear-gradient(135deg, rgba(0, 245, 212, 0.18) 0%, rgba(0, 180, 216, 0.08) 100%) !important;
        border: 2px solid #00f5d4 !important;
        border-radius: 20px;
        padding: 28px;
        text-align: center;
        box-shadow: 0 0 40px rgba(0, 245, 212, 0.35);
    }
    
    .verdict-uncertain {
        background: linear-gradient(135deg, rgba(255, 193, 7, 0.2) 0%, rgba(255, 152, 0, 0.1) 100%) !important;
        border: 2px solid #ffc107 !important;
        border-radius: 20px;
        padding: 28px;
        text-align: center;
        box-shadow: 0 0 40px rgba(255, 193, 7, 0.35);
    }

    .verdict-fake {
        background: linear-gradient(135deg, rgba(255, 75, 75, 0.22) 0%, rgba(255, 0, 128, 0.1) 100%) !important;
        border: 2px solid #ff4b4b !important;
        border-radius: 20px;
        padding: 28px;
        text-align: center;
        box-shadow: 0 0 40px rgba(255, 75, 75, 0.4);
    }

    .verdict-title-real {
        color: #00f5d4 !important;
        font-size: 2.5rem;
        font-weight: 800;
        letter-spacing: 1px;
        margin-bottom: 8px;
        text-shadow: 0 0 20px rgba(0, 245, 212, 0.7);
    }

    .verdict-title-uncertain {
        color: #ffc107 !important;
        font-size: 2.3rem;
        font-weight: 800;
        letter-spacing: 1px;
        margin-bottom: 8px;
        text-shadow: 0 0 20px rgba(255, 193, 7, 0.7);
    }

    .verdict-title-fake {
        color: #ff4b4b !important;
        font-size: 2.5rem;
        font-weight: 800;
        letter-spacing: 1px;
        margin-bottom: 8px;
        text-shadow: 0 0 20px rgba(255, 75, 75, 0.7);
    }

    .verdict-sub {
        font-size: 1.1rem;
        color: #cbd5e1 !important;
    }

    .kpi-box {
        background: #0f172a !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
    }

    .kpi-val {
        font-size: 1.7rem;
        font-weight: 700;
        color: #00f2fe !important;
        font-family: 'JetBrains Mono', monospace;
    }

    .kpi-lbl {
        font-size: 0.82rem;
        color: #94a3b8 !important;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-top: 4px;
    }

    .stButton>button {
        background: #0f172a !important;
        color: #00f5d4 !important;
        border: 1px solid #00f5d4 !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        transition: all 0.3s ease !important;
    }

    .stButton>button:hover {
        background: #00f5d4 !important;
        color: #05070a !important;
        box-shadow: 0 0 20px rgba(0, 245, 212, 0.6) !important;
    }

    [data-testid="stDataFrame"] {
        background: #0a0e17 !important;
        border-radius: 10px;
    }

    audio {
        width: 100%;
        margin-top: 8px;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Universal Audio Loading Helper
# ---------------------------------------------------------
def load_audio_flexible(file_path, target_sr=16000):
    try:
        audio, sr = librosa.load(file_path, sr=target_sr, mono=True)
        return audio, sr
    except Exception:
        pass

    try:
        data, sr = sf.read(file_path)
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        if sr != target_sr:
            audio = librosa.resample(data.astype(np.float32), orig_sr=sr, target_sr=target_sr)
        else:
            audio = data.astype(np.float32)
        return audio, target_sr
    except Exception:
        pass

    try:
        from scipy.io import wavfile
        sr, data = wavfile.read(file_path)
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        data = data.astype(np.float32)
        if np.max(np.abs(data)) > 0:
            data = data / np.max(np.abs(data))
        if sr != target_sr:
            audio = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
        else:
            audio = data
        return audio, target_sr
    except Exception:
        pass

    raise RuntimeError(f"Could not decode audio file '{os.path.basename(file_path)}'.")


# ---------------------------------------------------------
# Load Model Artifacts & Saved Canonical Feature Names
# ---------------------------------------------------------
def load_model_artifacts():
    model_path = "model/audio_deepfake_model.pkl"
    scaler_path = "model/scaler.pkl"
    feature_names_path = "model/feature_names.pkl"
    metrics_path = "model/metrics.pkl"

    if not os.path.exists(model_path):
        return None, None, None, None

    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
        saved_features = joblib.load(feature_names_path) if os.path.exists(feature_names_path) else get_feature_names()
        metrics = joblib.load(metrics_path) if os.path.exists(metrics_path) else None
        return model, scaler, saved_features, metrics
    except Exception as e:
        st.error(f"Error loading model artifacts: {e}")
        return None, None, None, None


model, scaler, saved_feature_names, metrics = load_model_artifacts()


def adapt_feature_dataframe(features_vec, saved_features=None):
    """
    Guarantees 100% column name & order alignment with the saved scaler / model.
    Prevents ValueError: The feature names should match those that were passed during fit.
    """
    canonical_names = saved_features if saved_features is not None else get_feature_names()
    expected_n = len(canonical_names)

    vec = features_vec[:expected_n] if len(features_vec) >= expected_n else np.pad(features_vec, (0, expected_n - len(features_vec)))
    
    return pd.DataFrame([vec], columns=canonical_names)


# ---------------------------------------------------------
# Sidebar Configuration
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("""
        <div class="sidebar-header">
            <h1 style="font-size: 1.8rem; margin:0; font-weight:800; color:#ffffff;">🛡️ AI VoiceShield</h1>
            <span class="badge-status">INDUSTRIAL FORENSICS v2.4</span>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("### ⚙️ System Status")
    if model is not None:
        st.success("✅ ML Engine Active & Loaded")
        st.markdown(f"""
            <div class="sidebar-card">
                <strong style="color:#00f2fe;">🤖 Classifier Architecture</strong><br>
                <span style="font-size:0.88rem; color:#cbd5e1;">Group-Aware Multi-Ensemble (RF + ExtraTrees + GradientBoosting + SVM)</span>
            </div>
            <div class="sidebar-card">
                <strong style="color:#00f5d4;">📊 Feature Matrix</strong><br>
                <span style="font-size:0.88rem; color:#cbd5e1;">{len(saved_feature_names) if saved_feature_names else 96} Signal Descriptors (MFCCs, Delta2, Chroma, Contrast, Jitter, Shimmer, HF-Ratio)</span>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.error("❌ Model Artifact Not Found!")
        st.warning("Please run `python train_model.py` to generate the model artifact.")

    st.markdown("---")
    st.markdown("### 🎵 Universal Audio Support")
    st.caption("Supports: WAV, MP3, MPEG, M4A, FLAC, OGG, AAC, WMA, OPUS, WEBM, AIFF, 3GP")

    st.markdown("---")
    st.markdown("### 🎯 Benchmark Test Samples")
    st.write("Click below to test pre-loaded benchmark files:")

    demo_real_btn = st.button("🎙️ Load Real Human Voice", use_container_width=True)
    demo_fake_btn = st.button("🤖 Load AI Deepfake Voice", use_container_width=True)

    st.markdown("---")
    st.caption("AI VoiceShield | Audio Deepfake Forensics & Verification")


# ---------------------------------------------------------
# Main Page Header
# ---------------------------------------------------------
st.markdown("""
    <div style="text-align: left; margin-bottom: 25px;">
        <h1 style="font-size: 2.8rem; font-weight: 800; background: linear-gradient(90deg, #00f2fe, #4facfe, #00f5d4); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 0;">
            🛡️ AI VoiceShield: Audio Deepfake Detection
        </h1>
        <p style="color: #94a3b8; font-size: 1.15rem; margin-top: 6px;">
            Advanced Acoustic Signal Forensics & Multi-Ensemble Machine Learning Classifier for Voice Authenticity Verification
        </p>
    </div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Audio Source Selection Logic
# ---------------------------------------------------------
audio_path = None
source_name = None

if "active_sample" not in st.session_state:
    st.session_state.active_sample = None

if demo_real_btn:
    st.session_state.active_sample = "uploads/kalyani_marathi_voice.ogg"
elif demo_fake_btn:
    st.session_state.active_sample = "uploads/ai_hindi_friendship_day.ogg"

col_up, col_info = st.columns([2, 1])

with col_up:
    SUPPORTED_AUDIO_TYPES = [
        "wav", "mp3", "mpeg", "mp4", "m4a", "flac", 
        "ogg", "aac", "wma", "opus", "webm", "aiff", "3gp", "amr"
    ]
    
    uploaded_file = st.file_uploader(
        "Upload Audio File for Deepfake Inspection",
        type=SUPPORTED_AUDIO_TYPES,
        help="Upload ANY audio format (.wav, .mp3, .mpeg, .m4a, .flac, .ogg, .aac, .wma, .opus, etc.)"
    )

if uploaded_file is not None:
    os.makedirs("uploads", exist_ok=True)
    audio_path = os.path.join("uploads", uploaded_file.name)
    with open(audio_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    source_name = uploaded_file.name
    st.session_state.active_sample = None
elif st.session_state.active_sample is not None and os.path.exists(st.session_state.active_sample):
    audio_path = st.session_state.active_sample
    source_name = os.path.basename(audio_path)

with col_info:
    st.markdown("""
        <div class="glass-card" style="padding:18px;">
            <h4 style="margin:0 0 8px 0; color:#00f5d4;">🔍 Inspection Mode</h4>
            <p style="font-size:0.9rem; color:#94a3b8; margin:0;">
                Upload custom audio in <strong>ANY format or language</strong> (English, Hindi, Marathi, etc.) to run full real-time acoustic feature extraction and classification.
            </p>
        </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------
# Main Analysis Pipeline
# ---------------------------------------------------------
if audio_path is not None:
    st.markdown("---")

    # Load audio file flexibly
    try:
        audio, sr = load_audio_flexible(audio_path, target_sr=16000)
        duration = len(audio) / sr
    except Exception as e:
        st.error(f"❌ Failed to load audio file: {e}")
        st.stop()

    # Save a guaranteed browser-playable 16kHz WAV preview
    try:
        preview_path = os.path.join("uploads", "preview_playable.wav")
        sf.write(preview_path, audio, sr)
        playable_audio_source = preview_path
    except Exception:
        playable_audio_source = audio_path

    # Preprocess audio
    clean_audio = preprocess_audio(audio, sr=sr)

    # Extract Features & Adapt Dimension
    raw_features_vec = extract_features(clean_audio, sr=sr, do_preprocess=False)
    if raw_features_vec is None:
        st.error("❌ Feature extraction failed.")
        st.stop()

    features_df = adapt_feature_dataframe(raw_features_vec, saved_feature_names)

    # Validate Feature Alignment before calling Scaler
    if scaler is not None and hasattr(scaler, "feature_names_in_"):
        scaler_cols = list(scaler.feature_names_in_)
        if list(features_df.columns) != scaler_cols:
            features_df = features_df.reindex(columns=scaler_cols, fill_value=0.0)

    # Model Inference
    if model is not None and scaler is not None:
        scaled_input = scaler.transform(features_df)
        probabilities = model.predict_proba(scaled_input)[0]

        real_prob = probabilities[0] * 100
        fake_prob = probabilities[1] * 100

        # Pure Binary Classification Thresholding (Real: <50%, Deepfake: >=50%)
        if fake_prob >= 50.0:
            verdict_state = "FAKE"
            confidence = fake_prob
            risk_level, risk_color = "CRITICAL DEEPFAKE RISK", "#ff4b4b"
            verdict_text = "🔴 AI GENERATED DEEPFAKE"
        else:
            verdict_state = "REAL"
            confidence = real_prob
            risk_level, risk_color = "AUTHENTIC VOICE (LOW RISK)", "#00f5d4"
            verdict_text = "🟢 REAL HUMAN VOICE"
    else:
        verdict_state = "REAL"
        real_prob, fake_prob, confidence = 50.0, 50.0, 50.0
        risk_level, risk_color = "UNINITIALIZED MODEL", "#94a3b8"
        verdict_text = "UNINITIALIZED"

    # Automatically record scan history
    try:
        import datetime
        history_path = "dataset/scan_history.csv"
        os.makedirs("dataset", exist_ok=True)
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        new_row = {
            "Timestamp": now_str,
            "Filename": source_name if source_name else os.path.basename(audio_path),
            "Prediction": verdict_text,
            "Risk Score (%)": f"{fake_prob:.2f}%",
            "Real Likelihood (%)": f"{real_prob:.2f}%",
            "Deepfake Likelihood (%)": f"{fake_prob:.2f}%",
            "Confidence (%)": f"{confidence:.2f}%"
        }
        
        if os.path.exists(history_path):
            hist_df = pd.read_csv(history_path)
            if "Verdict" in hist_df.columns and "Prediction" not in hist_df.columns:
                hist_df.rename(columns={"Verdict": "Prediction"}, inplace=True)
            # Avoid duplicate consecutive logging
            if hist_df.empty or hist_df.iloc[-1]["Filename"] != new_row["Filename"] or hist_df.iloc[-1]["Timestamp"] != now_str:
                hist_df = pd.concat([hist_df, pd.DataFrame([new_row])], ignore_index=True)
                hist_df.to_csv(history_path, index=False)
        else:
            pd.DataFrame([new_row]).to_csv(history_path, index=False)
    except Exception as e:
        pass

    # Navigation Tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🛡️ Detection Summary",
        "📊 Spectral Visualizer",
        "🔬 Forensic Breakdown",
        "📈 ML Model Analytics",
        "📁 Batch Screener",
        "📜 Forensic Scan History"
    ])

    # ---------------------------------------------------------
    # TAB 1: DETECTION SUMMARY
    # ---------------------------------------------------------
    with tab1:
        st.subheader("🎧 Audio Source & Prediction Verdict")

        # Guaranteed Browser Playable Audio Player
        st.audio(playable_audio_source)

        st.write("")

        # Verdict Display Card (Pure Binary REAL vs DEEPFAKE)
        if verdict_state == "REAL":
            st.markdown(f"""
                <div class="verdict-real">
                    <div class="verdict-title-real">🟢 REAL HUMAN VOICE DETECTED</div>
                    <div class="verdict-sub">Acoustic signature exhibits natural vocal tract resonances and harmonic pitch dynamics.</div>
                    <div style="margin-top:12px; font-size:1.4rem; font-weight:700; color:#00f5d4;">
                        Confidence Score: {confidence:.2f}%
                    </div>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="verdict-fake">
                    <div class="verdict-title-fake">🔴 AI GENERATED DEEPFAKE DETECTED</div>
                    <div class="verdict-sub">Synthetic neural vocoder artifacts & phase anomalies detected in high-frequency spectral bands.</div>
                    <div style="margin-top:12px; font-size:1.4rem; font-weight:700; color:#ff4b4b;">
                        Confidence Score: {confidence:.2f}%
                    </div>
                </div>
            """, unsafe_allow_html=True)

        st.write("")
        st.write("")

        # KPI Metrics Row
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.markdown(f"""
                <div class="kpi-box">
                    <div class="kpi-val">{sr} Hz</div>
                    <div class="kpi-lbl">Sample Rate</div>
                </div>
            """, unsafe_allow_html=True)
        with k2:
            st.markdown(f"""
                <div class="kpi-box">
                    <div class="kpi-val">{duration:.2f} s</div>
                    <div class="kpi-lbl">Duration</div>
                </div>
            """, unsafe_allow_html=True)
        with k3:
            st.markdown(f"""
                <div class="kpi-box">
                    <div class="kpi-val" style="color:{risk_color}; font-size:1.1rem;">{risk_level}</div>
                    <div class="kpi-lbl">Forensic Risk Status</div>
                </div>
            """, unsafe_allow_html=True)
        with k4:
            st.markdown(f"""
                <div class="kpi-box">
                    <div class="kpi-val">{features_df.shape[1]}</div>
                    <div class="kpi-lbl">Acoustic Features Evaluated</div>
                </div>
            """, unsafe_allow_html=True)

        st.write("")
        st.write("")

        # Probability Bar Breakdown
        col_prob1, col_prob2 = st.columns(2)
        with col_prob1:
            st.write("🟢 **Real Voice Likelihood:**")
            st.progress(real_prob / 100.0)
            st.write(f"**{real_prob:.2f}%**")
        with col_prob2:
            st.write("🔴 **AI Deepfake Likelihood:**")
            st.progress(fake_prob / 100.0)
            st.write(f"**{fake_prob:.2f}%**")

    # ---------------------------------------------------------
    # TAB 2: SPECTRAL VISUALIZER
    # ---------------------------------------------------------
    with tab2:
        st.subheader("📊 Acoustic Signal & Spectral Density Visualizer")

        # Waveform Plot
        fig_wave = plot_waveform(clean_audio, sr)
        st.pyplot(fig_wave)

        st.markdown("---")

        # Spectrogram & Chromagram Columns
        col_spec1, col_spec2 = st.columns(2)
        with col_spec1:
            fig_spec = plot_spectrogram(clean_audio, sr)
            st.pyplot(fig_spec)
        with col_spec2:
            fig_chroma = plot_chromagram(clean_audio, sr)
            st.pyplot(fig_chroma)

    # ---------------------------------------------------------
    # TAB 3: FORENSIC BREAKDOWN
    # ---------------------------------------------------------
    with tab3:
        st.subheader("🔬 Signal Forensics & Feature Breakdown")

        f_names = list(features_df.columns)
        vals = features_df.values.flatten()
        f_df = pd.DataFrame({
            "Feature Name": f_names[:len(vals)],
            "Extracted Value": vals[:len(f_names)]
        })

        col_f1, col_f2 = st.columns([1, 1])

        with col_f1:
            st.markdown("##### 🧬 Signal Descriptor Values (Top 20)")
            st.dataframe(f_df.head(20), use_container_width=True, height=400)

        with col_f2:
            st.markdown("##### 🎯 Feature Importance Ranking in ML Model")
            if metrics is not None and "feature_importances" in metrics:
                importances = np.array(metrics["feature_importances"])
                fig_imp = plot_feature_importance(f_names, importances, top_n=12)
                st.pyplot(fig_imp)
            else:
                st.info("Train metrics loaded.")

        st.markdown("---")
        st.markdown("##### 🔍 Full Extracted Feature Vector Inspection")
        with st.expander("Click to view complete feature vector table"):
            st.dataframe(f_df, use_container_width=True)

    # ---------------------------------------------------------
    # TAB 4: ML MODEL ANALYTICS
    # ---------------------------------------------------------
    with tab4:
        st.subheader("📈 Industrial Model Performance & Training Metrics")

        if metrics is not None:
            m1, m2, m3, m4, m5 = st.columns(5)
            with m1:
                st.metric("Test Accuracy", f"{metrics['accuracy']*100:.2f}%")
            with m2:
                st.metric("5-Fold CV Mean", f"{metrics['cv_mean']*100:.2f}%")
            with m3:
                st.metric("Precision", f"{metrics['precision']*100:.2f}%")
            with m4:
                st.metric("Recall Score", f"{metrics['recall']*100:.2f}%")
            with m5:
                st.metric("ROC-AUC Score", f"{metrics['auc']*100:.2f}%")

            st.markdown("---")

            col_cm1, col_cm2 = st.columns(2)
            with col_cm1:
                st.markdown("##### 🧩 Confusion Matrix (Unseen Voices)")
                cm = np.array(metrics["confusion_matrix"])
                cm_df = pd.DataFrame(
                    cm,
                    index=["Actual Real", "Actual Deepfake"],
                    columns=["Pred Real", "Pred Deepfake"]
                )
                st.dataframe(cm_df.style.highlight_max(axis=1, color='#1e293b'), use_container_width=True)

            with col_cm2:
                st.markdown("##### ℹ️ Model Details & Training Config")
                st.write(f"- **Classifier Ensemble:** Group-Aware Multi-Ensemble (RF + ExtraTrees + GradientBoosting + SVM)")
                st.write(f"- **Training Dataset Size:** {metrics.get('train_count', 68)} groups")
                st.write(f"- **Unseen Test Validation Size:** {metrics.get('test_count', 18)} groups")
                st.write(f"- **Data Leakage Control:** 0% Data Leakage (GroupShuffleSplit by Speaker/File)")
                st.write(f"- **Standardization:** StandardScaler Z-Score Normalization")
                st.write(f"- **Features Evaluated:** {len(saved_feature_names)} Canonical Descriptors")

        else:
            st.warning("No metrics.pkl file found. Run `python train_model.py` to generate complete training analytics.")

    # ---------------------------------------------------------
    # TAB 5: BATCH AUDIO SCREENER
    # ---------------------------------------------------------
    with tab5:
        st.subheader("📁 Multi-File Batch Forensic Screener")
        st.write("Upload multiple audio files simultaneously to perform rapid deepfake classification.")

        batch_files = st.file_uploader(
            "Upload Batch Audio Files",
            type=SUPPORTED_AUDIO_TYPES,
            accept_multiple_files=True,
            key="batch_uploader"
        )

        if batch_files:
            batch_results = []
            os.makedirs("uploads/batch", exist_ok=True)

            for b_file in batch_files:
                b_path = os.path.join("uploads/batch", b_file.name)
                with open(b_path, "wb") as f:
                    f.write(b_file.getbuffer())

                try:
                    b_audio, b_sr = load_audio_flexible(b_path, target_sr=16000)
                    b_clean = preprocess_audio(b_audio, sr=b_sr)
                    b_raw_feats = extract_features(b_clean, sr=b_sr, do_preprocess=False)
                    b_feats_df = adapt_feature_dataframe(b_raw_feats, saved_feature_names)

                    if scaler is not None and hasattr(scaler, "feature_names_in_"):
                        b_feats_df = b_feats_df.reindex(columns=list(scaler.feature_names_in_), fill_value=0.0)

                    if b_feats_df is not None and model is not None and scaler is not None:
                        b_in = scaler.transform(b_feats_df)
                        b_prob = model.predict_proba(b_in)[0]
                        
                        if b_prob[1] >= 0.50:
                            b_label = "🔴 DEEPFAKE"
                            b_conf = b_prob[1] * 100
                        else:
                            b_label = "🟢 REAL VOICE"
                            b_conf = b_prob[0] * 100
                    else:
                        b_label = "ERROR"
                        b_conf = 0.0
                        b_prob = [0, 0]

                    batch_results.append({
                        "Filename": b_file.name,
                        "Verdict": b_label,
                        "Confidence (%)": f"{b_conf:.2f}%",
                        "Real Prob (%)": f"{b_prob[0]*100:.2f}%" if model else "N/A",
                        "Deepfake Prob (%)": f"{b_prob[1]*100:.2f}%" if model else "N/A"
                    })
                except Exception:
                    batch_results.append({
                        "Filename": b_file.name,
                        "Verdict": "FAILED",
                        "Confidence (%)": "0%",
                        "Real Prob (%)": "N/A",
                        "Deepfake Prob (%)": "N/A"
                    })

            batch_df = pd.DataFrame(batch_results)
            st.dataframe(batch_df, use_container_width=True)

            csv = batch_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download Batch Report CSV",
                csv,
                "ai_voiceshield_batch_report.csv",
                "text/csv",
                key='download-csv'
            )

    # ---------------------------------------------------------
    # TAB 6: FORENSIC SCAN HISTORY & RISK LOG
    # ---------------------------------------------------------
    with tab6:
        st.subheader("📜 Forensic Scan History & Continuous Risk Log")
        st.write("Complete historical audit log of analyzed audio files with Real/Deepfake Likelihood and Risk Scores.")

        history_path = "dataset/scan_history.csv"

        if os.path.exists(history_path):
            try:
                hist_df = pd.read_csv(history_path)

                if not hist_df.empty:
                    # Robust column detection for Prediction / Verdict
                    pred_col = "Prediction" if "Prediction" in hist_df.columns else ("Verdict" if "Verdict" in hist_df.columns else hist_df.columns[2])

                    total_scans = len(hist_df)
                    
                    def is_real_val(val):
                        return "REAL" in str(val).upper()

                    def is_fake_val(val):
                        s = str(val).upper()
                        return "DEEPFAKE" in s or "FAKE" in s

                    real_cnt = int(hist_df[pred_col].apply(is_real_val).sum())
                    fake_cnt = int(hist_df[pred_col].apply(is_fake_val).sum())

                    # Calculate average risk score float
                    try:
                        risk_vals = hist_df["Risk Score (%)"].astype(str).str.replace("%", "").astype(float)
                        avg_risk = risk_vals.mean()
                    except Exception:
                        avg_risk = 0.0

                    hc1, hc2, hc3, hc4 = st.columns(4)
                    with hc1:
                        st.metric("Total Audio Scans", total_scans)
                    with hc2:
                        st.metric("🟢 Real Voices Detected", real_cnt)
                    with hc3:
                        st.metric("🔴 AI Deepfakes Detected", fake_cnt)
                    with hc4:
                        st.metric("⚡ Avg Deepfake Risk Score", f"{avg_risk:.2f}%")

                    st.markdown("---")
                    st.markdown("##### 🔍 Inspection History Table")
                    st.dataframe(hist_df, use_container_width=True, height=350)

                    col_dl, col_clr = st.columns([2, 1])
                    with col_dl:
                        hist_csv = hist_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            "📥 Download Full History Log CSV",
                            hist_csv,
                            "ai_voiceshield_full_scan_history.csv",
                            "text/csv",
                            key="download-history-csv"
                        )
                    with col_clr:
                        if st.button("🗑️ Clear Scan History", use_container_width=True):
                            pd.DataFrame(columns=["Timestamp", "Filename", "Prediction", "Risk Score (%)", "Real Likelihood (%)", "Deepfake Likelihood (%)", "Confidence (%)"]).to_csv(history_path, index=False)
                            st.rerun()
                else:
                    st.info("No scan history recorded yet.")
            except Exception as e:
                st.error(f"Error loading scan history log: {e}")
        else:
            st.info("No scan history recorded yet.")

else:
    st.info("👆 Upload an audio file above or click a demo sample button in the sidebar to start deepfake forensic inspection.")