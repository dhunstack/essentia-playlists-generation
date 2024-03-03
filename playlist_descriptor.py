import os
import streamlit as st
import pandas as pd
import random
from fileio import (
    get_saved_genre_activations,
    get_saved_file_paths,
    get_genre_names,
    get_saved_all_features,
)
from config import GENRE_DISCOGS_PATH, ALL_FEATURES_PATH, PLAYLISTS_DIR_PATH


class DescriptorPlaylist:
    def __init__(self):
        """
        Create a playlist based on audio analysis data.
        """
        self.genre_activations = self._load_genre_activations()
        self.all_features = self._load_all_features()
        self.tracks = list(self.genre_activations.index)

        self.create_sidebar()
        self.create_mainpage()
        self.results_handler()

    @st.cache_data
    def _load_genre_activations(_self):
        """
        Load the genre activations from the saved file.

        Returns:
            pd.DataFrame: DataFrame with genre activations
        """
        genre_activation = get_saved_genre_activations()
        file_paths = get_saved_file_paths()
        genre_names = get_genre_names()

        # Create a DataFrame with the genre activations
        genre_activation_df = pd.DataFrame(
            genre_activation, columns=genre_names, index=file_paths
        )
        return genre_activation_df

    @st.cache_data
    def _load_all_features(_self):
        """
        Load all features from the saved file.

        Returns:
            pd.DataFrame: DataFrame with all features
        """
        all_features = get_saved_all_features()
        file_paths = get_saved_file_paths()
        all_features_df = pd.DataFrame(all_features, index=file_paths)

        # Extract the key and scale from the key_edma field
        all_features_df["key"] = all_features_df["key_krumhansl"].apply(
            lambda x: x.split(" ")[0]
        )
        all_features_df["scale"] = all_features_df["key_krumhansl"].apply(
            lambda x: x.split(" ")[1]
        )

        return all_features_df

    def create_sidebar(self):
        """
        Create the sidebar with the search filters.

        Returns:
            None
        """
        st.sidebar.write("# 🔍 Filter parameters")
        self.tempo_slider = st.sidebar.slider(
            f"BPM",
            value=[0, 250],
        )
        self.voice_instrumental_select = st.sidebar.multiselect(
            "Voice/Instrumental",
            ["Voice", "Instrumental"],
        )
        self.danceability_slider = st.sidebar.slider(f"Danceability", value=[0.0, 1.0])
        self.arousal_slider = st.sidebar.slider(f"Arousal", value=[0.0, 9.0])
        self.valence_slider = st.sidebar.slider(f"Valence", value=[0.0, 9.0])
        self.key_select = st.sidebar.multiselect(
            "Key",
            ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"],
        )
        self.scale_select = st.sidebar.multiselect(
            "Scale",
            ["major", "minor"],
        )

    def create_mainpage(self):
        """
        Create the main page with the genre activations and ranking.

        Returns:
            None
        """
        # Title and Description
        st.write("# Audio analysis playlists")
        st.write(f"Using genre activations from `{GENRE_DISCOGS_PATH}`.")
        st.write(f"Using features from `{ALL_FEATURES_PATH}`.")

        # Select genre activations
        genre_names = self.genre_activations.columns
        st.write("Loaded audio analysis for", len(self.genre_activations), "tracks.")
        st.write("## 🔍 Select by Genre")

        self.genre_select = st.multiselect("Select by genre activations:", genre_names)

        if self.genre_select:
            # Show the distribution of activation values for the selected styles.
            st.write(self.genre_activations[self.genre_select].describe())

            genre_select_str = ", ".join(self.genre_select)
            self.genre_activation_range = st.slider(
                f"Select tracks with at least one of `{genre_select_str}` activations within range:",
                value=[0.2, 1.0],
            )

        # Rank by genre activations
        st.write("## 🔝 Rank")
        self.genre_rank = st.multiselect(
            "Rank by genre activations (multiplies activations for selected genres):",
            genre_names,
            [],
        )

        # Post-process to limit the number of tracks and add shuffle
        st.write("## 🔀 Post-process")
        self.max_tracks = st.number_input(
            "Maximum number of tracks (0 for all):", min_value=0, step=1, value=0
        )
        self.shuffle = st.checkbox("Random shuffle")

    def results_handler(self):
        """
        Handle the results and display the playlist.

        Returns:
            None
        """
        st.write("## 🔊 Results")
        self.process_genre_fields()
        self.process_sidebar_fields()
        self.post_process()
        self.display_playlist()
        self.save_playlist()

    def process_genre_fields(self):
        """
        Process the genre fields to filter the tracks.

        Returns:
            None
        """
        if self.genre_select:
            audio_analysis_query = self.genre_activations.loc[self.tracks][
                self.genre_select
            ]

            # Extract all tracks that have at least one genre activation within the selected range.
            audio_analysis_query = audio_analysis_query[
                (audio_analysis_query >= self.genre_activation_range[0]).any(axis=1)
                & (audio_analysis_query <= self.genre_activation_range[1]).any(axis=1)
            ]
            self.tracks = list(audio_analysis_query.index)

        if self.genre_rank:
            audio_analysis_query = self.genre_activations.loc[self.tracks][
                self.genre_rank
            ]
            audio_analysis_query["RANK"] = audio_analysis_query[self.genre_rank[0]]
            for style in self.genre_rank[1:]:
                audio_analysis_query["RANK"] *= audio_analysis_query[style]
            ranked = audio_analysis_query.sort_values(["RANK"], ascending=[False])
            ranked = ranked[["RANK"] + self.genre_rank]
            self.tracks = list(ranked.index)

            st.write("Applied ranking by audio style predictions.")
            st.write(ranked)

    def process_sidebar_fields(self):
        """
        Process the sidebar fields to filter the tracks.

        Returns:
            None
        """

        # Filter by tempo
        if self.tempo_slider:
            audio_analysis_query = self.all_features.loc[self.tracks]["tempo"]

            audio_analysis_query = audio_analysis_query[
                (audio_analysis_query >= self.tempo_slider[0])
                & (audio_analysis_query <= self.tempo_slider[1])
            ]
            self.tracks = list(audio_analysis_query.index)

        # Filter by danceability
        if self.danceability_slider:
            audio_analysis_query = self.all_features.loc[self.tracks][
                "danceability_probability"
            ]

            audio_analysis_query = audio_analysis_query[
                (audio_analysis_query >= self.danceability_slider[0])
                & (audio_analysis_query <= self.danceability_slider[1])
            ]
            self.tracks = list(audio_analysis_query.index)

        # Filter by arousal
        if self.arousal_slider:
            audio_analysis_query = self.all_features.loc[self.tracks]["arousal"]

            audio_analysis_query = audio_analysis_query[
                (audio_analysis_query >= self.arousal_slider[0])
                & (audio_analysis_query <= self.arousal_slider[1])
            ]
            self.tracks = list(audio_analysis_query.index)

        # Filter by valence
        if self.valence_slider:
            audio_analysis_query = self.all_features.loc[self.tracks]["valence"]

            audio_analysis_query = audio_analysis_query[
                (audio_analysis_query >= self.valence_slider[0])
                & (audio_analysis_query <= self.valence_slider[1])
            ]
            self.tracks = list(audio_analysis_query.index)

        # Filter by voice/instrumental
        if self.voice_instrumental_select:
            audio_analysis_query = self.all_features.loc[self.tracks][
                "instrumental_probability"
            ]

            if "Voice" in self.voice_instrumental_select:
                audio_analysis_query = audio_analysis_query[audio_analysis_query <= 0.5]
            if "Instrumental" in self.voice_instrumental_select:
                audio_analysis_query = audio_analysis_query[audio_analysis_query > 0.5]
            if (
                "Voice" in self.voice_instrumental_select
                and "Instrumental" in self.voice_instrumental_select
            ):
                audio_analysis_query = self.all_features.loc[self.tracks]

            self.tracks = list(audio_analysis_query.index)

        # Filter by key
        if self.key_select:
            audio_analysis_query = self.all_features.loc[self.tracks]["key"]

            audio_analysis_query = audio_analysis_query[
                audio_analysis_query.isin(self.key_select)
            ]
            self.tracks = list(audio_analysis_query.index)

        # Filter by scale
        if self.scale_select:
            audio_analysis_query = self.all_features.loc[self.tracks]["scale"]

            audio_analysis_query = audio_analysis_query[
                audio_analysis_query.isin(self.scale_select)
            ]
            self.tracks = list(audio_analysis_query.index)

    def post_process(self):
        """
        Post-process the tracks to limit the number and add shuffle.

        Returns:
            None
        """

        if self.max_tracks:
            self.tracks = self.tracks[: self.max_tracks]
            st.write("Using top", len(self.tracks), "tracks from the results.")

        if self.shuffle:
            random.shuffle(self.tracks)
            st.write("Applied random shuffle.")

    def display_playlist(self):
        """
        Display the playlist with audio previews.

        Returns:
            None
        """
        st.write(f"Total tracks in the playlist: {len(self.tracks)}")
        display_tracks = self.tracks[:10]
        st.write(f"Audio previews for the first {len(display_tracks)} results:")
        for mp3 in display_tracks:
            st.audio(mp3, format="audio/mp3", start_time=0)

    def save_playlist(self):
        """
        Save the playlist to a file.

        Returns:
            None
        """
        # Add a button for saving the playlist
        st.write("## 💾 Save")
        self.playlist_name = st.text_input("Playlist name:", "descriptor_playlist")
        if st.button("Save"):
            playlist_path = os.path.join(
                PLAYLISTS_DIR_PATH, f"{self.playlist_name}.m3u8"
            )

            with open(playlist_path, "w") as f:
                # Modify relative mp3 paths to make them accessible from the playlist folder.
                mp3_paths = [os.path.join("..", mp3) for mp3 in self.tracks]

                f.write("\n".join(mp3_paths))
                st.write(f"Stored M3U playlist (local filepaths) to `{playlist_path}`.")


def main():
    descriptor_playlist = DescriptorPlaylist()


if __name__ == "__main__":
    main()
