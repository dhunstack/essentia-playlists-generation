import essentia.standard as estd
import numpy as np
from config import (
    TEMPOCNN_MODEL_PATH,
    DISCOGS_EMBEDDINGS_MODEL_PATH,
    MSD_EMBEDDINGS_MODEL_PATH,
    GENRE_DISCOGS_MODEL_PATH,
    VOICE_DISCOGS_MODEL_PATH,
    DANCEABILITY_DISCOGS_MODEL_PATH,
    AROUSAL_MUSICNN_MODEL_PATH,
)
from fileio import get_genre_names

# Create instances of the algorithms
tempo_model = estd.TempoCNN(graphFilename=TEMPOCNN_MODEL_PATH)
key_temperley_model = estd.KeyExtractor(profileType="temperley")
key_krumhansl_model = estd.KeyExtractor(profileType="krumhansl")
key_edma_model = estd.KeyExtractor(profileType="edma")
loudness_model = estd.LoudnessEBUR128()
embeddings_discogs_model = estd.TensorflowPredictEffnetDiscogs(
    graphFilename=DISCOGS_EMBEDDINGS_MODEL_PATH, output="PartitionedCall:1"
)
embeddings_msd_model = estd.TensorflowPredictMusiCNN(
    graphFilename=MSD_EMBEDDINGS_MODEL_PATH, output="model/dense/BiasAdd"
)
genre_model = estd.TensorflowPredict2D(
    graphFilename=GENRE_DISCOGS_MODEL_PATH,
    input="serving_default_model_Placeholder",
    output="PartitionedCall:0",
)
instrumental_model = estd.TensorflowPredict2D(
    graphFilename=VOICE_DISCOGS_MODEL_PATH, output="model/Softmax"
)
danceability_model = estd.TensorflowPredict2D(
    graphFilename=DANCEABILITY_DISCOGS_MODEL_PATH, output="model/Softmax"
)
arousal_valence_model = estd.TensorflowPredict2D(
    graphFilename=AROUSAL_MUSICNN_MODEL_PATH, output="model/Identity"
)

# Load the genre classes
genre_classes = get_genre_names()


def get_tempo(resampled_11k_audio):
    """
    Returns the tempo of the audio

    Args:
        resampled_11k_audio (np array): Mono audio resampled to 11,025 Hz

    Returns:
        int : Tempo of the audio in BPM
    """

    global_tempo, _, _ = tempo_model(resampled_11k_audio)
    return global_tempo


def get_key_temperley(mono_audio):
    """
    Returns the key of the audio using the temperley model

    Args:
        mono_audio (np array): Mono audio with a sample rate of 44,100 Hz

    Returns:
        str : Key of the audio
    """

    key, scale, _ = key_temperley_model(mono_audio)
    return key + " " + scale


def get_key_krumhansl(mono_audio):
    """
    Returns the key of the audio using the krumhansl model

    Args:
        mono_audio (np array): Mono audio with a sample rate of 44,100 Hz

    Returns:
        str : Key of the audio
    """

    key, scale, _ = key_krumhansl_model(mono_audio)
    return key + " " + scale


def get_key_edma(mono_audio):
    """
    Returns the key of the audio using the edma model

    Args:
        mono_audio (np array): Mono audio with a sample rate of 44,100 Hz

    Returns:
        str : Key of the audio
    """

    key, scale, _ = key_edma_model(mono_audio)
    return key + " " + scale


def get_loudness(stereo_audio):
    """
    Returns the loudness of the audio

    Args:
        stereo_audio (np array): Stereo audio with a sample rate of 44,100 Hz

    Returns:
        float : Loudness of the audio in LUFS
    """

    _, _, integrated_loudness, _ = loudness_model(stereo_audio)
    return integrated_loudness


def get_discogs_embeddings(resampled_16k_audio):
    """
    Returns the discogs embeddings of the audio

    Args:
        resampled_16k_audio (np array): Mono audio resampled to 16,000 Hz

    Returns:
        np array : Discogs embeddings of the audio
    """

    embeddings = embeddings_discogs_model(resampled_16k_audio)
    return embeddings


def get_msd_embeddings(resampled_16k_audio):
    """
    Returns the msd embeddings of the audio

    Args:
        resampled_16k_audio (np array): Mono audio resampled to 16,000 Hz

    Returns:
        np array : MSD embeddings of the audio
    """

    embeddings = embeddings_msd_model(resampled_16k_audio)
    return embeddings


def get_genre_activations(discogs_embeddings):
    """
    Returns the genre activations of the audio

    Args:
        discogs_embeddings (np array): Discogs embeddings of the audio

    Returns:
        np array : Genre activations of the audio for 400 discogs genre categories
    """

    activations = np.mean(genre_model(discogs_embeddings), axis=0)
    return activations


def get_instrumental_probability(discogs_embeddings):
    """
    Returns the instrumental probability of the audio

    Args:
        discogs_embeddings (np array): Discogs embeddings of the audio

    Returns:
        float : Instrumental probability of the audio
    """

    instrumental_probability = np.mean(instrumental_model(discogs_embeddings), axis=0)[
        0
    ]
    return instrumental_probability


def get_danceability_probability(discogs_embeddings):
    """
    Returns the danceability probability of the audio

    Args:
        discogs_embeddings (np array): Discogs embeddings of the audio

    Returns:
        float : Danceability probability of the audio
    """

    danceability_probability = np.mean(danceability_model(discogs_embeddings), axis=0)[
        0
    ]
    return danceability_probability


def get_valence_arousal(msd_embeddings):
    """
    Returns the valence and arousal of the audio

    Args:
        msd_embeddings (np array): MSD embeddings of the audio

    Returns:
        float : Valence of the audio
        float : Arousal of the audio
    """

    valence, arousal = np.mean(arousal_valence_model(msd_embeddings), axis=0)
    return valence, arousal


def get_audio_features(stereo_audio, mono_audio, resampled_11k_audio):
    """
    Returns the audio features

    Args:
        stereo_audio (np array): Stereo audio with a sample rate of 44,100 Hz
        mono_audio (np array): Mono audio with a sample rate of 44,100 Hz
        resampled_11k_audio (np array): Mono audio resampled to 11,025 Hz

    Returns:
        dict : Audio features
    """

    tempo = get_tempo(resampled_11k_audio)
    key_temperley = get_key_temperley(mono_audio)
    key_krumhansl = get_key_krumhansl(mono_audio)
    key_edma = get_key_edma(mono_audio)
    loudness = get_loudness(stereo_audio)
    return {
        "tempo": tempo,
        "key_temperley": key_temperley,
        "key_krumhansl": key_krumhansl,
        "key_edma": key_edma,
        "loudness": loudness,
    }


def get_embeddings(resampled_16k_audio):
    """
    Returns the embeddings of the audio

    Args:
        resampled_16k_audio (np array): Mono audio resampled to 16,000 Hz

    Returns:
        np array : Discogs embeddings of the audio
        np array : MSD embeddings of the audio
    """

    discogs_embeddings = get_discogs_embeddings(resampled_16k_audio)
    msd_embeddings = get_msd_embeddings(resampled_16k_audio)
    return discogs_embeddings, msd_embeddings


def get_genre_distribution(discogs_embeddings):
    """
    Returns the genre distribution of the audio and the most probable genre

    Args:
        discogs_embeddings (np array): Discogs embeddings of the audio

    Returns:
        np array : Genre activations of the audio for 400 discogs genre categories
        dict : Most probable genre of the audio
    """

    genre_activations = get_genre_activations(discogs_embeddings)

    genre_index = np.argmax(genre_activations)
    genre_feature = {"genre": genre_classes[genre_index]}

    return genre_activations, genre_feature


def get_embeddings_features(discogs_embeddings, msd_embeddings):
    """
    Returns the embeddings features

    Args:
        discogs_embeddings (np array): Discogs embeddings of the audio
        msd_embeddings (np array): MSD embeddings of the audio

    Returns:
        dict : Embeddings features
    """

    instrumental_probability = get_instrumental_probability(discogs_embeddings)
    danceability_probability = get_danceability_probability(discogs_embeddings)
    valence, arousal = get_valence_arousal(msd_embeddings)
    return {
        "instrumental_probability": instrumental_probability,
        "danceability_probability": danceability_probability,
        "arousal": arousal,
        "valence": valence,
    }
