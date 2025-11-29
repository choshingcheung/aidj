"""
Data loading utilities for Spotify Million Playlist Dataset.
"""

import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import numpy as np
from .config import RAW_DATA_DIR, PROCESSED_DATA_DIR, ModelConfig


class PlaylistDataLoader:
    """
    Loads and preprocesses Spotify Million Playlist Dataset.
    """

    def __init__(self, raw_data_dir=None, processed_data_dir=None):
        """
        Initialize data loader.

        Args:
            raw_data_dir: Directory containing raw JSON files
            processed_data_dir: Directory for saving processed data
        """
        self.raw_data_dir = Path(raw_data_dir or RAW_DATA_DIR)
        self.processed_data_dir = Path(processed_data_dir or PROCESSED_DATA_DIR)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)

    def load_raw_playlists(self, max_files=None, sample_playlists=None):
        """
        Load raw playlists from JSON files.

        Args:
            max_files: Maximum number of JSON files to load (None = all)
            sample_playlists: Number of playlists to sample (None = all)

        Returns:
            list: List of playlist dictionaries
        """
        # Find all slice files
        slice_files = sorted(self.raw_data_dir.glob("**/mpd.slice.*.json"))

        if not slice_files:
            raise FileNotFoundError(
                f"No playlist files found in {self.raw_data_dir}. "
                "Please download and extract the Spotify Million Playlist Dataset."
            )

        if max_files:
            slice_files = slice_files[:max_files]

        print(f"Found {len(slice_files)} slice files")

        # Load playlists
        all_playlists = []

        for slice_file in tqdm(slice_files, desc="Loading playlists"):
            with open(slice_file, 'r') as f:
                data = json.load(f)
                all_playlists.extend(data['playlists'])

        print(f"Loaded {len(all_playlists)} total playlists")

        # Sample if requested
        if sample_playlists and sample_playlists < len(all_playlists):
            np.random.seed(ModelConfig.RANDOM_SEED)
            indices = np.random.choice(len(all_playlists), sample_playlists, replace=False)
            all_playlists = [all_playlists[i] for i in indices]
            print(f"Sampled {len(all_playlists)} playlists")

        return all_playlists

    def filter_playlists(self, playlists, min_length=None, max_length=None):
        """
        Filter playlists by length.

        Args:
            playlists: List of playlist dictionaries
            min_length: Minimum number of tracks
            max_length: Maximum number of tracks

        Returns:
            list: Filtered playlists
        """
        min_length = min_length or ModelConfig.MIN_PLAYLIST_LENGTH
        max_length = max_length or ModelConfig.MAX_PLAYLIST_LENGTH

        filtered = [
            p for p in playlists
            if min_length <= p['num_tracks'] <= max_length
        ]

        print(f"Filtered from {len(playlists)} to {len(filtered)} playlists "
              f"(length {min_length}-{max_length})")

        return filtered

    def extract_track_info(self, playlists):
        """
        Extract track information from playlists.

        Args:
            playlists: List of playlist dictionaries

        Returns:
            tuple: (playlist_df, track_df)
                - playlist_df: DataFrame with playlist metadata
                - track_df: DataFrame with unique tracks
        """
        playlist_data = []
        track_sequences = []
        all_tracks = {}

        for i, playlist in enumerate(tqdm(playlists, desc="Extracting tracks")):
            # Playlist metadata
            playlist_data.append({
                'playlist_id': i,
                'name': playlist.get('name', ''),
                'num_tracks': playlist['num_tracks'],
                'num_followers': playlist.get('num_followers', 0),
                'num_edits': playlist.get('num_edits', 0)
            })

            # Track sequence
            sequence = []
            for track in playlist['tracks']:
                track_uri = track['track_uri']
                sequence.append(track_uri)

                # Collect unique track info
                if track_uri not in all_tracks:
                    all_tracks[track_uri] = {
                        'track_uri': track_uri,
                        'track_name': track['track_name'],
                        'artist_name': track['artist_name'],
                        'album_name': track['album_name'],
                        'duration_ms': track.get('duration_ms', 0)
                    }

            track_sequences.append(sequence)

        # Create DataFrames
        playlist_df = pd.DataFrame(playlist_data)
        playlist_df['track_sequence'] = track_sequences

        track_df = pd.DataFrame.from_dict(all_tracks, orient='index')
        track_df.reset_index(drop=True, inplace=True)

        print(f"Extracted {len(playlist_df)} playlists and {len(track_df)} unique tracks")

        return playlist_df, track_df

    def compute_track_frequencies(self, playlist_df):
        """
        Compute how often each track appears.

        Args:
            playlist_df: DataFrame with track_sequence column

        Returns:
            pd.Series: Track URI to frequency mapping
        """
        from collections import Counter

        all_tracks = []
        for sequence in playlist_df['track_sequence']:
            all_tracks.extend(sequence)

        frequencies = Counter(all_tracks)
        return pd.Series(frequencies)

    def filter_rare_tracks(self, playlist_df, track_df, min_frequency=None):
        """
        Remove tracks that appear too rarely.

        Args:
            playlist_df: Playlist DataFrame
            track_df: Track DataFrame
            min_frequency: Minimum number of appearances

        Returns:
            tuple: (filtered_playlist_df, filtered_track_df)
        """
        min_frequency = min_frequency or ModelConfig.MIN_SONG_FREQUENCY

        # Compute frequencies
        freq = self.compute_track_frequencies(playlist_df)

        # Filter tracks
        valid_tracks = set(freq[freq >= min_frequency].index)
        print(f"Keeping {len(valid_tracks)} tracks (appeared >= {min_frequency} times)")

        # Filter playlists
        filtered_sequences = []
        valid_playlists = []

        for idx, row in playlist_df.iterrows():
            filtered_seq = [t for t in row['track_sequence'] if t in valid_tracks]

            # Keep playlist if it still has enough tracks
            if len(filtered_seq) >= ModelConfig.MIN_PLAYLIST_LENGTH:
                filtered_sequences.append(filtered_seq)
                valid_playlists.append(idx)

        filtered_playlist_df = playlist_df.loc[valid_playlists].copy()
        filtered_playlist_df['track_sequence'] = filtered_sequences
        filtered_playlist_df['num_tracks'] = [len(s) for s in filtered_sequences]
        filtered_playlist_df.reset_index(drop=True, inplace=True)

        filtered_track_df = track_df[track_df['track_uri'].isin(valid_tracks)].copy()
        filtered_track_df.reset_index(drop=True, inplace=True)

        print(f"After filtering: {len(filtered_playlist_df)} playlists, "
              f"{len(filtered_track_df)} unique tracks")

        return filtered_playlist_df, filtered_track_df

    def train_val_test_split(self, playlist_df):
        """
        Split playlists into train/val/test sets.

        Args:
            playlist_df: Playlist DataFrame

        Returns:
            tuple: (train_df, val_df, test_df)
        """
        np.random.seed(ModelConfig.RANDOM_SEED)

        n = len(playlist_df)
        indices = np.random.permutation(n)

        train_size = int(n * ModelConfig.TRAIN_RATIO)
        val_size = int(n * ModelConfig.VAL_RATIO)

        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]

        train_df = playlist_df.iloc[train_indices].copy().reset_index(drop=True)
        val_df = playlist_df.iloc[val_indices].copy().reset_index(drop=True)
        test_df = playlist_df.iloc[test_indices].copy().reset_index(drop=True)

        print(f"Split: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test")

        return train_df, val_df, test_df

    def save_processed_data(self, playlist_df, track_df, suffix=""):
        """
        Save processed data to disk.

        Args:
            playlist_df: Playlist DataFrame
            track_df: Track DataFrame
            suffix: Optional suffix for filenames
        """
        playlist_file = self.processed_data_dir / f"playlists{suffix}.pkl"
        track_file = self.processed_data_dir / f"tracks{suffix}.pkl"

        playlist_df.to_pickle(playlist_file)
        track_df.to_pickle(track_file)

        print(f"Saved to {playlist_file} and {track_file}")

    def load_processed_data(self, suffix=""):
        """
        Load processed data from disk.

        Args:
            suffix: Optional suffix for filenames

        Returns:
            tuple: (playlist_df, track_df)
        """
        playlist_file = self.processed_data_dir / f"playlists{suffix}.pkl"
        track_file = self.processed_data_dir / f"tracks{suffix}.pkl"

        if not playlist_file.exists() or not track_file.exists():
            raise FileNotFoundError(f"Processed data not found. Run preprocessing first.")

        playlist_df = pd.read_pickle(playlist_file)
        track_df = pd.read_pickle(track_file)

        print(f"Loaded {len(playlist_df)} playlists and {len(track_df)} tracks")

        return playlist_df, track_df


def preprocess_full_pipeline(max_files=None, num_playlists=None):
    """
    Run full preprocessing pipeline.

    Args:
        max_files: Max number of JSON files to load
        num_playlists: Number of playlists to sample

    Returns:
        dict: All processed DataFrames
    """
    loader = PlaylistDataLoader()

    # Load raw data
    print("\n=== Loading Raw Data ===")
    playlists = loader.load_raw_playlists(
        max_files=max_files,
        sample_playlists=num_playlists or ModelConfig.NUM_PLAYLISTS
    )

    # Filter by length
    print("\n=== Filtering by Length ===")
    playlists = loader.filter_playlists(playlists)

    # Extract track info
    print("\n=== Extracting Track Info ===")
    playlist_df, track_df = loader.extract_track_info(playlists)

    # Filter rare tracks
    print("\n=== Filtering Rare Tracks ===")
    playlist_df, track_df = loader.filter_rare_tracks(playlist_df, track_df)

    # Train/val/test split
    print("\n=== Splitting Data ===")
    train_df, val_df, test_df = loader.train_val_test_split(playlist_df)

    # Save
    print("\n=== Saving Processed Data ===")
    loader.save_processed_data(playlist_df, track_df, suffix="_all")
    loader.save_processed_data(train_df, track_df, suffix="_train")
    loader.save_processed_data(val_df, track_df, suffix="_val")
    loader.save_processed_data(test_df, track_df, suffix="_test")

    print("\n✓ Preprocessing complete!")

    return {
        'playlist_all': playlist_df,
        'track_all': track_df,
        'train': train_df,
        'val': val_df,
        'test': test_df
    }


if __name__ == "__main__":
    # Example usage: process first 10 files
    print("Running preprocessing pipeline...")
    data = preprocess_full_pipeline(max_files=10, num_playlists=10000)

    print("\nSummary:")
    print(f"Total playlists: {len(data['playlist_all'])}")
    print(f"Unique tracks: {len(data['track_all'])}")
    print(f"Train playlists: {len(data['train'])}")
    print(f"Val playlists: {len(data['val'])}")
    print(f"Test playlists: {len(data['test'])}")
