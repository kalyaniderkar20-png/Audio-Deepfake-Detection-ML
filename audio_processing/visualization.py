import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

# Set dark background style globally for Matplotlib plots
plt.style.use('dark_background')


def plot_waveform(audio, sample_rate):
    """
    Plots audio time-domain waveform with modern cyber-glow styling.
    """
    fig, ax = plt.subplots(figsize=(10, 3.5), facecolor='#05070a')
    ax.set_facecolor('#05070a')

    time_axis = np.linspace(0, len(audio) / sample_rate, num=len(audio))
    
    # Plot glow aura effect
    ax.plot(time_axis, audio, color='#00f2fe', alpha=0.3, linewidth=2)
    # Plot crisp foreground wave
    ax.plot(time_axis, audio, color='#00f5d4', alpha=0.95, linewidth=0.8)

    ax.set_title("Audio Amplitude Waveform (Time Domain)", color='#ffffff', fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel("Time (Seconds)", color='#a0aec0', fontsize=10)
    ax.set_ylabel("Amplitude", color='#a0aec0', fontsize=10)
    ax.tick_params(colors='#a0aec0')
    ax.grid(True, color='#1e293b', linestyle='--', alpha=0.5)

    plt.tight_layout()
    return fig


def plot_spectrogram(audio, sample_rate):
    """
    Plots Mel Spectrogram using power-to-dB conversion and magma colormap.
    """
    fig, ax = plt.subplots(figsize=(10, 4), facecolor='#05070a')
    ax.set_facecolor('#05070a')

    # Calculate Mel Spectrogram
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_mels=128,
        fmax=sample_rate // 2
    )

    # Convert to Decibels (dB scale) - CRITICAL FIX
    mel_db = librosa.power_to_db(mel, ref=np.max)

    img = librosa.display.specshow(
        mel_db,
        sr=sample_rate,
        x_axis="time",
        y_axis="mel",
        fmax=sample_rate // 2,
        ax=ax,
        cmap="magma"
    )

    ax.set_title("Log Mel Spectrogram (Frequency & Energy Density)", color='#ffffff', fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel("Time (Seconds)", color='#a0aec0', fontsize=10)
    ax.set_ylabel("Frequency (Hz)", color='#a0aec0', fontsize=10)
    ax.tick_params(colors='#a0aec0')

    cbar = fig.colorbar(img, ax=ax, format="%+2.0f dB")
    cbar.ax.yaxis.set_tick_params(color='#a0aec0')
    plt.setp(plt.getp(cbar.ax, 'yticklabels'), color='#a0aec0')
    cbar.set_label('Energy (dB)', color='#a0aec0')

    plt.tight_layout()
    return fig


def plot_chromagram(audio, sample_rate):
    """
    Plots Chromagram STFT showing pitch intensity across 12 pitch classes.
    """
    fig, ax = plt.subplots(figsize=(10, 3.5), facecolor='#05070a')
    ax.set_facecolor('#05070a')

    chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate)

    img = librosa.display.specshow(
        chroma,
        sr=sample_rate,
        x_axis='time',
        y_axis='chroma',
        ax=ax,
        cmap='cool'
    )

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
    """
    Plots horizontal bar chart of top ML feature importances.
    """
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
