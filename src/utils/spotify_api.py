"""
Spotify API utilities for fetching audio features.
"""

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
import json
from pathlib import Path
from tqdm import tqdm
import time
from .config import SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET, CACHE_DIR

class SpotifyFeatureFetcher:
    """
    Fetches audio features for tracks from Spotify API with caching.
    """

    def __init__(self, client_id=None, client_secret=None, cache_dir=None):
        """
        Initialize Spotify API client.

        Args:
            client_id: Spotify API client ID
            client_secret: Spotify API client secret
            cache_dir: Directory for caching API responses
        """
        self.client_id = client_id or SPOTIFY_CLIENT_ID
        self.client_secret = client_secret or SPOTIFY_CLIENT_SECRET
        self.cache_dir = Path(cache_dir or CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize Spotify client
        auth_manager = SpotifyClientCredentials(
            client_id=self.client_id,
            client_secret=self.client_secret
        )
        self.sp = spotipy.Spotify(auth_manager=auth_manager)

        # Load cache
        self.cache_file = self.cache_dir / "audio_features_cache.json"
        self.cache = self._load_cache()

    def _load_cache(self):
        """Load cached audio features."""
        if self.cache_file.exists():
            with open(self.cache_file, 'r') as f:
                return json.load(f)
        return {}

    def _save_cache(self):
        """Save audio features cache."""
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache, f)

    def get_audio_features(self, track_uri):
        """
        Get audio features for a single track.

        Args:
            track_uri: Spotify track URI (e.g., 'spotify:track:6rqhFgbbKwnb9MLmUQDhG6')

        Returns:
            dict: Audio features or None if not found
        """
        # Check cache first
        if track_uri in self.cache:
            return self.cache[track_uri]

        try:
            # Extract track ID from URI
            track_id = track_uri.split(':')[-1]

            # Fetch from API
            features = self.sp.audio_features(track_id)[0]

            if features:
                # Cache the result
                self.cache[track_uri] = features
                self._save_cache()

            return features

        except Exception as e:
            print(f"Error fetching features for {track_uri}: {e}")
            return None

    def get_audio_features_batch(self, track_uris, batch_size=50):
        """
        Get audio features for multiple tracks with batching.

        Args:
            track_uris: List of Spotify track URIs
            batch_size: Number of tracks to fetch per API call (max 100)

        Returns:
            dict: Mapping from track URI to audio features
        """
        results = {}
        uncached_uris = [uri for uri in track_uris if uri not in self.cache]

        # First, get all cached results
        for uri in track_uris:
            if uri in self.cache:
                results[uri] = self.cache[uri]

        if not uncached_uris:
            return results

        # Fetch uncached tracks in batches
        print(f"Fetching {len(uncached_uris)} tracks from Spotify API...")

        for i in tqdm(range(0, len(uncached_uris), batch_size)):
            batch = uncached_uris[i:i + batch_size]
            track_ids = [uri.split(':')[-1] for uri in batch]

            try:
                features_list = self.sp.audio_features(track_ids)

                for uri, features in zip(batch, features_list):
                    if features:
                        results[uri] = features
                        self.cache[uri] = features

                # Rate limiting
                time.sleep(0.1)

            except Exception as e:
                print(f"Error fetching batch: {e}")
                continue

        # Save updated cache
        self._save_cache()

        return results

    def features_to_dataframe(self, features_dict):
        """
        Convert features dictionary to pandas DataFrame.

        Args:
            features_dict: Dictionary mapping track URIs to features

        Returns:
            pd.DataFrame: Audio features with track URI as index
        """
        df = pd.DataFrame.from_dict(features_dict, orient='index')
        df.index.name = 'track_uri'
        return df

    def get_relevant_features(self, features):
        """
        Extract only the relevant features for our model.

        Args:
            features: Full audio features dict from Spotify

        Returns:
            dict: Relevant features only
        """
        if not features:
            return None

        relevant_keys = [
            'tempo',  # BPM
            'key',
            'mode',
            'energy',
            'valence',
            'danceability',
            'acousticness',
            'instrumentalness',
            'loudness',
            'speechiness',
            'liveness',
            'time_signature',
            'duration_ms'
        ]

        return {k: features.get(k) for k in relevant_keys if k in features}


if __name__ == "__main__":
    # Test the API
    fetcher = SpotifyFeatureFetcher()

    # Example track URI (replace with actual track)
    test_uri = "spotify:track:6rqhFgbbKwnb9MLmUQDhG6"

    print("Testing Spotify API...")
    features = fetcher.get_audio_features(test_uri)

    if features:
        print("\nAudio Features:")
        for key, value in fetcher.get_relevant_features(features).items():
            print(f"  {key}: {value}")
    else:
        print("Failed to fetch features. Check your API credentials.")
