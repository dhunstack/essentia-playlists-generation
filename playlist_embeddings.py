import os
import streamlit as st
import numpy as np
import random
from fileio import (
    get_saved_file_paths,
    get_saved_discogs_embeddings,
    get_saved_msd_embeddings,
)
from config import DISCOGS_EMBEDDINGS_PATH, MSD_EMBEDDINGS_PATH, PLAYLISTS_DIR_PATH


class EmbeddingPlaylist:
    def __init__(self):
        """
        Create a playlist based on audio analysis data.
        """
        self.discogs_embeddings_similarity = self._cosine_similarity("discogs")
        self.msd_embeddings_similarity = self._cosine_similarity("msd")
        self.all_tracks = self._load_file_paths()

        self.create_page()
        self.results_handler()
    
    @st.cache_data
    def _load_file_paths(_self):
        """
        Load the file paths from the saved file.

        Returns:
            list: The file paths.
        """

        file_paths = get_saved_file_paths()
        return file_paths
    
    @st.cache_data
    def _cosine_similarity(_self, embedding_name):
        """
        Compute the cosine similarity between the embedding vectors.
        
        Args:
            embedding_name (str): The name of the embedding to use.
            
        Returns:
            np.array: The cosine similarity matrix.
        """

        if embedding_name == "discogs":
            embeddings_func = get_saved_discogs_embeddings
        elif embedding_name == "msd":
            embeddings_func = get_saved_msd_embeddings
        else:
            raise ValueError("Invalid embedding name.")
        
        A = embeddings_func()
        Anorm = A / np.linalg.norm(A, axis=1)[:, np.newaxis]
        return np.dot(Anorm, Anorm.T)

    def create_page(self):
        # Title and Description
        st.write("# Track similarity playlists")
        st.write(f"Using discogs embeddings from `{DISCOGS_EMBEDDINGS_PATH}`.")
        st.write(f"Using msd embeddings from `{MSD_EMBEDDINGS_PATH}`.")

        st.write("Loaded audio analysis for", len(self.all_tracks), "tracks.")
        st.write("## 🔍 Generate playlists by similarity")

        with st.sidebar:
            # Select track to generate playlists
            st.write("### Select a track to generate playlists")
            self.track_select = st.selectbox("Select the track", self.all_tracks, index=None)

            # Display the selected track
            if self.track_select:
                st.write("### Selected Track")
                st.audio(self.track_select, format="audio/mp3", start_time=0)

            # Select the number of tracks for the playlist
            st.write("### Select number of required tracks for the playlists")
            self.playlist_length = st.number_input(
                "Number of tracks(0 for all)", min_value=0, max_value=100, value=10
            )   

    def results_handler(self):
        """
        Handle the results and display the playlist.

        Returns:
            None
        """
        if self.track_select:
            st.write("## 🔊 Results")
            col1, col2 = st.columns(2)
            with col1:
                self.process_discogs()
            with col2:
                self.process_msd()

    def process_discogs(self):
        """
        Process the discogs embeddings and display the playlist.

        Returns:
            None
        """
        st.write("### Discogs Playlist")
        self.top_discogs_similar_tracks = self.find_similar_tracks("discogs")
        self.display_playlist(self.top_discogs_similar_tracks)
        self.save_discogs_playlist()

    def process_msd(self):
        """
        Process the MSD embeddings and display the playlist.

        Returns:
            None
        """
        st.write("### MSD Playlist")
        self.top_msd_similar_tracks = self.find_similar_tracks("msd")
        self.display_playlist(self.top_msd_similar_tracks)
        self.save_msd_playlist()

    def find_similar_tracks(self, embedding_name):
        """
        Get the top similar tracks to the selected track using the embeddings.

        Args:
            embedding_name (str): The name of the embedding to use.

        Returns:
            list: The top similar tracks.
        """
        if embedding_name == "discogs":
            similarity_matrix = self.discogs_embeddings_similarity
        elif embedding_name == "msd":
            similarity_matrix = self.msd_embeddings_similarity
        else:
            raise ValueError("Invalid embedding name.")
        
        track_index = self.all_tracks.index(self.track_select)
        top_similar_indexes = np.argsort(similarity_matrix[track_index])[::-1][1: self.playlist_length+1]

        return [self.all_tracks[i] for i in top_similar_indexes]

    def display_playlist(self, playlist):
        """
        Display the playlist with audio previews.

        Returns:
            None
        """
        st.write(f"Total tracks in the playlist: {len(playlist)}")
        display_tracks = playlist[:10]
        st.write(f"Audio previews for the first {len(display_tracks)} results:")
        for mp3 in display_tracks:
            st.audio(mp3, format="audio/mp3", start_time=0)

    def save_discogs_playlist(self):
        """
        Save the discogs playlist to a file.

        Returns:
            None
        """
        # Add a button for saving the playlist
        st.write("#### 💾 Save Discogs Playlist")
        self.discogs_playlist_name = st.text_input("Discogs Playlist name:", "discogs_embeddings_playlist")
        if st.button("Save Discogs Playlist"):
            playlist_path = os.path.join(
                PLAYLISTS_DIR_PATH, f"{self.discogs_playlist_name}.m3u8"
            )

            with open(playlist_path, "w") as f:
                # Modify relative mp3 paths to make them accessible from the playlist folder.
                mp3_paths = [os.path.join("..", mp3) for mp3 in self.top_discogs_similar_tracks]

                f.write("\n".join(mp3_paths))
                st.write(f"Stored discogs playlist to `{playlist_path}`.")

    def save_msd_playlist(self):
        """
        Save the msd playlist to a file.

        Returns:
            None
        """
        # Add a button for saving the playlist
        st.write("#### 💾 Save MSD Playlist")
        self.msd_playlist_name = st.text_input("MSD Playlist name:", "msd_embeddings_playlist")
        if st.button("Save MSD Playlist"):
            playlist_path = os.path.join(
                PLAYLISTS_DIR_PATH, f"{self.msd_playlist_name}.m3u8"
            )

            with open(playlist_path, "w") as f:
                # Modify relative mp3 paths to make them accessible from the playlist folder.
                mp3_paths = [os.path.join("..", mp3) for mp3 in self.top_msd_similar_tracks]

                f.write("\n".join(mp3_paths))
                st.write(f"Stored msd playlist to `{playlist_path}`.")


def main():
    embedding_playlist = EmbeddingPlaylist()


if __name__ == "__main__":
    main()
