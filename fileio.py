import json
import pickle
import os
import numpy as np
from config import (
    DISCOGS_EMBEDDINGS_PATH,
    MSD_EMBEDDINGS_PATH,
    GENRE_DISCOGS_PATH,
    ALL_FEATURES_PATH,
    FILE_PATHS_PATH,
    DISCOGS_EMBEDDINGS_METADATA_PATH,
    FEATURES_DIR_PATH,
    WEIGHTS_DIR_PATH,
    METADATA_DIR_PATH,
    RESULTS_DIR_PATH,
    PLAYLISTS_DIR_PATH,
)

def create_dir_if_not_exist(directory):
    """
    Create directory if it does not exist

    Args:
        directory (str): Directory path

    Returns:
        None
    """

    return os.makedirs(directory) if not os.path.exists(directory) else None

create_dir_if_not_exist(FEATURES_DIR_PATH)
create_dir_if_not_exist(WEIGHTS_DIR_PATH)
create_dir_if_not_exist(METADATA_DIR_PATH)
create_dir_if_not_exist(RESULTS_DIR_PATH)
create_dir_if_not_exist(PLAYLISTS_DIR_PATH)

def load_json(json_path):
    """
    Load JSON file as dictionary

    Args:
        json_path (str): Path to the JSON file

    Returns:
        dict: JSON file as dictionary
    """

    with open(json_path) as f:
        data = json.load(f)
    return data


def get_genre_names():
    """
    Get genre classes

    Returns:
        list: List of genre classes
    """

    return load_json(DISCOGS_EMBEDDINGS_METADATA_PATH)["classes"]


def open_files():
    """
    Open all file objects and return the file objects

    Returns:
        tuple: Tuple of file objects
    """

    discogs_embeddings_file = open(DISCOGS_EMBEDDINGS_PATH, "wb")
    msd_embeddings_file = open(MSD_EMBEDDINGS_PATH, "wb")
    genre_discogs_file = open(GENRE_DISCOGS_PATH, "wb")
    all_features_file = open(ALL_FEATURES_PATH, "wb")
    file_paths_file = open(FILE_PATHS_PATH, "wb")
    return (
        discogs_embeddings_file,
        msd_embeddings_file,
        genre_discogs_file,
        all_features_file,
        file_paths_file,
    )


def dump_pickle(file, data):
    """
    Dump data to pickle file

    Args:
        file (file object): File object to write to
        data (object): Data to write to file

    Returns:
        None
    """

    pickle.dump(data, file)


def close_files(
    discogs_embeddings_file,
    msd_embeddings_file,
    genre_discogs_file,
    all_features_file,
    file_paths_file,
):
    """
    Close all file objects

    Args:
        discogs_embeddings_file (file object): File object
        msd_embeddings_file (file object): File object
        genre_discogs_file (file object): File object
        all_features_file (file object): File object
        file_paths_file (file object): File object

    Returns:
        None
    """

    discogs_embeddings_file.close()
    msd_embeddings_file.close()
    genre_discogs_file.close()
    all_features_file.close()
    file_paths_file.close()
    return


def load_pickled(file_path):
    """
    Given a file path, load pickled objects until the end of the file

    Args:
        file_path (str): Path to the pickled file

    Returns:
        list: List of pickled objects
    """

    res = []
    with open(file_path, "rb") as file:
        while True:
            try:
                res.append(pickle.load(file))
            except EOFError:
                break
    return res


def get_saved_discogs_embeddings():
    """
    Return a 2D numpy array with Discogs embeddings for each audio file

    Returns:
        np array: Discogs embeddings for each audio file
    """

    return np.array(load_pickled(DISCOGS_EMBEDDINGS_PATH))


def get_saved_msd_embeddings():
    """
    Return a 2D numpy array with MSD embeddings for each audio file

    Returns:
        np array: MSD embeddings for each audio file
    """

    return np.array(load_pickled(MSD_EMBEDDINGS_PATH))


def get_saved_genre_activations():
    """
    Return a 2D numpy array with genre activations for each audio file

    Returns:
        np array: Genre activations for each audio file
    """

    return np.array(load_pickled(GENRE_DISCOGS_PATH))


def get_saved_all_features():
    """
    Return a list of dictionaries

    Returns:
        list: List of dictionaries
    """

    return load_pickled(ALL_FEATURES_PATH)


def get_saved_file_paths():
    """
    Return a list of file paths

    Returns:
        list: List of file paths
    """

    return load_pickled(FILE_PATHS_PATH)
