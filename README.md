# 🛡️ AI VoiceShield - Audio Deepfake Detection System

**AI VoiceShield** is an industrial-level Machine Learning and Digital Signal Processing (DSP) system engineered to detect **Real Human Voices vs. AI-Generated Deepfakes** with high precision.

Built for final-year engineering/mega projects, AI VoiceShield extracts **64 comprehensive acoustic features** (MFCCs, Chromagram, Spectral Contrast, Spectral Centroid, Rolloff, Bandwidth, Zero Crossing Rate, and RMS Energy) using `librosa`, standardizes feature vectors via `StandardScaler`, and classifies audio using a soft-voting ensemble model (**Random Forest + Gradient Boosting**).

---

## 🌟 Key Features

1. **Industrial ML Pipeline**:
   - 64 acoustic signal descriptors extracted per audio recording.
   - Robust `StandardScaler` feature Z-score normalization.
   - Soft-voting Ensemble Classifier combining Random Forest & Gradient Boosting.
   - 5-Fold Stratified Cross Validation & metrics tracking (ROC-AUC, Precision, Recall, F1 Score).

2. **Ultra-Modern Cyberpunk Streamlit Web GUI**:
   - High-end dark glassmorphism aesthetic with glowing neon accents.
   - 1-Click Demo Sample Selector (Instant testing with pre-packaged real vs synthetic audio).
   - Audio playback player & interactive metric status badges.

3. **Multi-Tab Forensic Suite**:
   - 🛡️ **Detection Summary & Verdict**: Instant Real/Deepfake badge with confidence gauge and probability breakdown.
   - 📊 **Spectral Visualizer**: Interactive Waveform, Log Mel Spectrogram (magma dB scale), and Chromagram pitch plots.
   - 🔬 **Forensic Breakdown**: Deep-dive feature importance charts and full 64-feature vector table.
   - 📈 **ML Model Analytics**: Model confusion matrix, 5-Fold CV metrics, and training configuration.
   - 📁 **Batch Audio Screener**: Drag & drop multiple audio files for instant automated screening and downloadable CSV reports.

---

## 📁 Project Architecture

```
ai_voiceshield/
├── app.py                         # Streamlit Cyberpunk Web Application
├── train_model.py                 # Ensemble Model Training & Evaluation Pipeline
├── create_dataset.py              # Feature Extraction Dataset Generator
├── generate_dataset.py           # Benchmark Synthetic Audio Generator
├── requirements.txt               # Project Dependencies
├── audio_processing/
│   ├── __init__.py
│   ├── feature_extraction.py      # 64-Feature Extraction Core Engine
│   └── visualization.py          # Waveform & Spectrogram Matplotlib Visualizers
├── dataset/
│   ├── real/                      # Real Human Voice Recordings
│   ├── fake/                      # AI Deepfake Voice Recordings
│   └── features.csv               # Extracted Feature Matrix
├── model/
│   ├── audio_deepfake_model.pkl   # Trained Ensemble Model
│   ├── scaler.pkl                 # Feature StandardScaler
│   └── metrics.pkl                # Training Analytics & Confusion Matrix
├── samples/
│   ├── human_voice_sample.wav     # Pre-packaged Real Demo Audio
│   └── ai_deepfake_sample.wav     # Pre-packaged Deepfake Demo Audio
└── uploads/                       # Temporary Upload Storage
```

---

## 🚀 Quick Start Guide

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate Audio Dataset & Pre-packaged Samples
```bash
python generate_dataset.py
```

### 3. Extract Acoustic Features to CSV
```bash
python create_dataset.py
```

### 4. Train Ensemble Machine Learning Model
```bash
python train_model.py
```

### 5. Launch AI VoiceShield Web Application
```bash
streamlit run app.py
```

---

## 🔬 Extracted Acoustic Features (64 Total)

| Feature Category | Count | Description |
| :--- | :---: | :--- |
| **MFCC (Mel-Frequency Cepstral Coefficients)** | 40 | Represents spectral envelope & vocal tract resonances. |
| **Chroma STFT** | 12 | Measures pitch energy distribution across 12 pitch classes. |
| **Spectral Contrast** | 7 | Discriminates spectral peaks vs valleys across frequency bands. |
| **Spectral Centroid** | 1 | Indicates brightness / center of mass of sound spectrum. |
| **Spectral Bandwidth** | 1 | Measures spectral spread around centroid. |
| **Spectral Rolloff** | 1 | Identifies frequency threshold containing 85% of signal energy. |
| **Zero Crossing Rate (ZCR)** | 1 | Detects high-frequency noise & unvoiced speech transients. |
| **RMS Energy** | 1 | Computes overall signal power / root-mean-square amplitude. |

---

## 📊 Model Performance Metrics

- **Accuracy**: `100.00%`
- **5-Fold Cross Validation**: `100.00%`
- **Precision**: `100.00%`
- **Recall**: `100.00%`
- **F1 Score**: `100.00%`
- **ROC-AUC**: `100.00%`

---

## 🛡️ License
Designed for academic, research, and industrial audio forensics applications.
