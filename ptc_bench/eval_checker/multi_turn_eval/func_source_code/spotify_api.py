from copy import deepcopy
from typing import Dict, List, Optional, Union

DEFAULT_STATE = {
    "playlists": {},
    "tracks": {},
    "recently_played": [],
    "saved_playlists": [],
    "user_playlists": [],
    "playlist_counter": 0,
    "track_counter": 0,
}


class SpotifyAPI:
    """
    A class representing the Spotify API for music streaming services.

    This class provides methods for searching tracks and playlists, managing
    playlists, getting recommendations, and viewing listening history.
    """

    def __init__(self):
        self.playlists: Dict[str, Dict]
        self.tracks: Dict[str, Dict]
        self.recently_played: List[str]
        self.saved_playlists: List[str]
        self.user_playlists: List[str]
        self.playlist_counter: int
        self.track_counter: int
        self._api_description = "This tool belongs to the SpotifyAPI, which provides functionality for music streaming including searching tracks, managing playlists, and getting personalized recommendations."

    def _load_scenario(self, scenario: dict, long_context=False) -> None:
        DEFAULT_STATE_COPY = deepcopy(DEFAULT_STATE)
        self.playlists = scenario.get("playlists", DEFAULT_STATE_COPY["playlists"])
        self.tracks = scenario.get("tracks", DEFAULT_STATE_COPY["tracks"])
        self.recently_played = scenario.get("recently_played", DEFAULT_STATE_COPY["recently_played"])
        self.saved_playlists = scenario.get("saved_playlists", DEFAULT_STATE_COPY["saved_playlists"])
        self.user_playlists = scenario.get("user_playlists", DEFAULT_STATE_COPY["user_playlists"])
        self.playlist_counter = scenario.get("playlist_counter", DEFAULT_STATE_COPY["playlist_counter"])
        self.track_counter = scenario.get("track_counter", DEFAULT_STATE_COPY["track_counter"])

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, SpotifyAPI):
            return False
        for attr_name in vars(self):
            if attr_name.startswith("_"):
                continue
            if getattr(self, attr_name) != getattr(value, attr_name):
                return False
        return True

    def search_tracks(self, query: str, limit: int = 5) -> Dict[str, List[Dict]]:
        """
        Search for tracks by name or artist.

        Args:
            query (str): Search query string.
            limit (int): [Optional] Maximum number of results. Default 5.
        Returns:
            tracks (List[Dict]): List of matching track dictionaries.
        """
        query_lower = query.lower()
        matches = []
        for track in self.tracks.values():
            if (query_lower in track["name"].lower() or
                    query_lower in track.get("artist", "").lower()):
                matches.append(track)
        return {"tracks": matches[:limit]}

    def search_playlists(self, query: str, limit: int = 5) -> Dict[str, List[Dict]]:
        """
        Search for playlists by name or description.

        Args:
            query (str): Search query string.
            limit (int): [Optional] Maximum number of results. Default 5.
        Returns:
            playlists (List[Dict]): List of matching playlist dictionaries.
        """
        query_lower = query.lower()
        matches = []
        for pl in self.playlists.values():
            if (query_lower in pl["name"].lower() or
                    query_lower in pl.get("description", "").lower()):
                matches.append(pl)
        return {"playlists": matches[:limit]}

    def get_playlist(self, playlist_id: str) -> Dict:
        """
        Get details of a specific playlist including its tracks.

        Args:
            playlist_id (str): ID of the playlist to retrieve.
        Returns:
            id (str): Playlist ID.
            name (str): Playlist name.
            description (str): Playlist description.
            tracks (List[Dict]): List of track dictionaries in the playlist.
        """
        if playlist_id not in self.playlists:
            return {"error": f"Playlist '{playlist_id}' not found."}
        pl = self.playlists[playlist_id]
        resolved_tracks = [
            self.tracks[tid] for tid in pl.get("tracks", []) if tid in self.tracks
        ]
        return {
            "id": pl["id"],
            "name": pl["name"],
            "description": pl.get("description", ""),
            "tracks": resolved_tracks,
        }

    def create_playlist(self, name: str, description: str = "") -> Dict[str, str]:
        """
        Create a new playlist.

        Args:
            name (str): Name of the new playlist.
            description (str): [Optional] Description of the playlist.
        Returns:
            playlist_id (str): ID of the created playlist.
            name (str): Playlist name.
            description (str): Playlist description.
            tracks (List): Empty list of tracks.
        """
        self.playlist_counter += 1
        pl_id = f"pl_new_{self.playlist_counter}"
        playlist = {
            "id": pl_id,
            "name": name,
            "description": description,
            "tracks": [],
        }
        self.playlists[pl_id] = playlist
        self.user_playlists.append(pl_id)
        return {
            "playlist_id": pl_id,
            "name": name,
            "description": description,
            "tracks": [],
        }

    def add_track_to_playlist(self, playlist_id: str, track_id: str) -> Dict[str, bool]:
        """
        Add a track to a playlist.

        Args:
            playlist_id (str): ID of the playlist.
            track_id (str): ID of the track to add.
        Returns:
            add_status (bool): True if the track was added successfully.
        """
        if playlist_id not in self.playlists:
            return {"error": f"Playlist '{playlist_id}' not found."}
        if track_id not in self.tracks:
            return {"error": f"Track '{track_id}' not found."}
        if track_id in self.playlists[playlist_id]["tracks"]:
            return {"add_status": False}
        self.playlists[playlist_id]["tracks"].append(track_id)
        return {"add_status": True}

    def remove_track_from_playlist(self, playlist_id: str, track_id: str) -> Dict[str, bool]:
        """
        Remove a track from a playlist.

        Args:
            playlist_id (str): ID of the playlist.
            track_id (str): ID of the track to remove.
        Returns:
            remove_status (bool): True if the track was removed successfully.
        """
        if playlist_id not in self.playlists:
            return {"error": f"Playlist '{playlist_id}' not found."}
        if track_id not in self.playlists[playlist_id]["tracks"]:
            return {"remove_status": False}
        self.playlists[playlist_id]["tracks"].remove(track_id)
        return {"remove_status": True}

    def get_track_details(self, track_id: str) -> Dict:
        """
        Get details of a specific track.

        Args:
            track_id (str): ID of the track.
        Returns:
            id (str): Track ID.
            name (str): Track name.
            artist (str): Artist name.
            album (str): Album name.
            duration_ms (int): Duration in milliseconds.
        """
        if track_id not in self.tracks:
            return {"error": f"Track '{track_id}' not found."}
        return self.tracks[track_id]

    def get_recently_played(self, limit: int = 10) -> Dict[str, List[Dict]]:
        """
        Get the user's recently played tracks.

        Args:
            limit (int): [Optional] Maximum number of tracks to return. Default 10.
        Returns:
            tracks (List[Dict]): List of recently played track dictionaries.
        """
        track_ids = self.recently_played[:limit]
        result = [self.tracks[tid] for tid in track_ids if tid in self.tracks]
        return {"tracks": result}

    def get_user_playlists(self) -> Dict[str, List[Dict]]:
        """
        Get the current user's playlists.

        Returns:
            playlists (List[Dict]): List of the user's playlist dictionaries.
        """
        result = [
            self.playlists[pid] for pid in self.user_playlists if pid in self.playlists
        ]
        return {"playlists": result}

    def get_recommendations(
        self, seed_track_ids: List[str], limit: int = 5
    ) -> Dict[str, List[Dict]]:
        """
        Get track recommendations based on seed tracks.

        Args:
            seed_track_ids (List[str]): List of track IDs to use as seeds.
            limit (int): [Optional] Maximum number of recommendations. Default 5.
        Returns:
            tracks (List[Dict]): List of recommended track dictionaries.
        """
        # Tracks not in any playlist and not in seeds
        all_playlist_tracks = set()
        for pl in self.playlists.values():
            all_playlist_tracks.update(pl.get("tracks", []))

        seed_set = set(seed_track_ids)
        recommendations = [
            track for tid, track in self.tracks.items()
            if tid not in all_playlist_tracks and tid not in seed_set
        ]
        return {"tracks": recommendations[:limit]}

    def save_playlist(self, playlist_id: str) -> Dict[str, bool]:
        """
        Save a playlist to the user's library.

        Args:
            playlist_id (str): ID of the playlist to save.
        Returns:
            save_status (bool): True if saved successfully, False if already saved.
        """
        if playlist_id not in self.playlists:
            return {"error": f"Playlist '{playlist_id}' not found."}
        if playlist_id in self.saved_playlists:
            return {"save_status": False}
        self.saved_playlists.append(playlist_id)
        return {"save_status": True}
