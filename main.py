from pathlib import Path
from audio import load_audio
from features import (
    get_audio_features,
    get_embeddings,
    get_genre_distribution,
    get_embeddings_features,
)
from fileio import open_files, close_files, dump_pickle
from config import DATA_PATH
import essentia
from tqdm import tqdm

# Deactivate the warnings
essentia.log.warningActive = False


def main():
    """
    Main function to process all the mp3 files in the directory

    Returns:
        None
    """
    # Generator for all mp3 files in the directory
    pathlist = Path(DATA_PATH).rglob("*.mp3")

    # Open all file objects
    (
        discogs_embeddings_file,
        msd_embeddings_file,
        genre_discogs_file,
        all_features_file,
        file_paths_file,
    ) = open_files()

    # Loop through all the files
    for path in tqdm(pathlist, total=2100):
        # Convert Path Object to String
        path_in_str = str(path)

        try:
            # Load audio as stereo, mono, resampled 16kHz and resampled 11kHz
            stereo, mono, resampled_16k, resampled_11k = load_audio(path_in_str)

            # Get features
            audio_features = get_audio_features(stereo, mono, resampled_11k)
            discogs_embeddings, msd_embeddings = get_embeddings(resampled_16k)
            genre_activations, genre_feature = get_genre_distribution(
                discogs_embeddings
            )
            embeddings_features = get_embeddings_features(
                discogs_embeddings, msd_embeddings
            )
            all_features = audio_features | genre_feature | embeddings_features
        except Exception as e:
            print(f"Error processing {path_in_str}: {str(e)}")
            continue

        # Compute average embeddings before saving
        discogs_embeddings = discogs_embeddings.mean(axis=0)
        msd_embeddings = msd_embeddings.mean(axis=0)

        # Save the features to the files
        dump_pickle(discogs_embeddings_file, discogs_embeddings)
        dump_pickle(msd_embeddings_file, msd_embeddings)
        dump_pickle(genre_discogs_file, genre_activations)
        dump_pickle(all_features_file, all_features)
        dump_pickle(file_paths_file, path_in_str)

    # Close all file objects
    close_files(
        discogs_embeddings_file,
        msd_embeddings_file,
        genre_discogs_file,
        all_features_file,
        file_paths_file,
    )


if __name__ == "__main__":
    main()

# TODO
# - Shift pathlib to fileio.py
# - Support resuming file processing from last file
# - Shift tqdm code to fileio.py
# - Error handling for fileio.py
