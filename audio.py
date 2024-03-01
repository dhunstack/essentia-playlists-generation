import essentia.standard as estd

# Create instances of the algorithms
mono_mixer = estd.MonoMixer()
resample_16k = estd.Resample(outputSampleRate=16000)
resample_11k = estd.Resample(outputSampleRate=11025)


def load_stereo_audio(audio_path):
    """
    Load stereo audio

    Args:
        audio_path (str): Path to the audio file

    Returns:
        np array : Stereo audio
        int : Sample rate
        int : Number of channels
    """
    stereo, sr, nc, _, _, _ = estd.AudioLoader(filename=audio_path)()
    return stereo, sr, nc


def get_mono_audio(stereo, nc):
    """
    Downmix to mono

    Args:
        stereo (np array): Stereo audio
        nc (int): Number of channels

    Returns:
        np array : Mono audio
    """
    mono = mono_mixer(stereo, nc)
    return mono


def get_resampled_16k_audio(mono):
    """
    Resample to 16kHz

    Args:
        mono (np array): Mono audio

    Returns:
        np array : Resampled audio
    """
    resampled = resample_16k(mono)
    return resampled


def get_resampled_11k_audio(mono):
    """
    Resample to 11kHz

    Args:
        mono (np array): Mono audio

    Returns:
        np array : Resampled audio
    """
    resampled = resample_11k(mono)
    return resampled


def load_audio(audio_path):
    """
    Load audio as stereo, mono, resampled 16kHz and resampled 11kHz

    Args:
        audio_path (str): Path to the audio file

    Returns:
        np array : Stereo audio
        np array : Mono audio
        np array : Resampled audio at 16kHz
        np array : Resampled audio at 11kHz
    """
    stereo, sr, nc = load_stereo_audio(audio_path)
    mono = get_mono_audio(stereo, nc)
    resampled_16k = get_resampled_16k_audio(mono)
    resampled_11k = get_resampled_11k_audio(mono)
    return stereo, mono, resampled_16k, resampled_11k
