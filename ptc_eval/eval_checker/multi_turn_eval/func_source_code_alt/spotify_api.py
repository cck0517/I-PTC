"""Variant of SpotifyAPI used by the intermediate experiment.

Two transient-failure injection points:

  - ``get_track_popularity_index`` — adds the popularity-ranking method
    used by the original 5 yelp/spotify intermediate queries.
  - ``get_playlist`` — overrides the per-playlist getter used by the
    user-playlist-aggregation queries (track ubiquity, artist reach,
    duplicate detection, etc.).

Per-call failure thresholds come from the task's ``flakiness_config``
via ``check_flaky``; absent that, the legacy default of fail-on-13th-call
applies (preserves behavior for tasks that pre-date explicit config).

The class is named ``SpotifyAPI`` so that tracebacks and repr strings
surfaced to the model do not broadcast that this is a variant; the module
lives at ``func_source_code_alt/spotify_api.py`` and is selected via the
``SpotifyAPIAlt`` key in CLASS_FILE_PATH_MAPPING.
"""

from typing import Any, Dict

from ..func_source_code.spotify_api import SpotifyAPI as _BaseSpotifyAPI
from ._flaky import check_flaky


class SpotifyError(Exception):
    def __init__(self, error_code: str, message: str, **context: Any) -> None:
        super().__init__(message)
        self.error = {"error_code": error_code, "message": message, **context}

    def __str__(self) -> str:  # noqa: D401
        return f"SpotifyError: {self.args[0]}"


class SpotifyAPI(_BaseSpotifyAPI):
    def get_track_popularity_index(self, track_id: str) -> Dict[str, Any]:
        """
        Get the popularity index for a track this week.

        The popularity index is a score from roughly 10 to 100 computed from
        the track's play count and listener metrics; higher values indicate
        more popular tracks this week.

        Args:
            track_id (str): The track to look up.

        Returns:
            Dict[str, Any]: track_id, name, artist, popularity_index.
        """
        if track_id not in self.tracks:
            raise SpotifyError(
                "TRACK_NOT_FOUND",
                f"Track '{track_id}' not found.",
                track_id=track_id,
            )
        track = self.tracks[track_id]

        if check_flaky(self, "get_track_popularity_index"):
            raise SpotifyError(
                "INTERNAL_ERROR",
                "Upstream popularity service returned an internal error. Please try again.",
                track_id=track_id,
            )

        # Score = play_count (capped at 500k) / 10k + listener_rating * 10.
        # Keeps values roughly in [10, 100].
        play_count = int(track.get("play_count", 0) or 0)
        listener_rating = float(track.get("listener_rating", 0) or 0)
        score = round(min(play_count, 500_000) / 10_000 + listener_rating * 10, 1)
        return {
            "track_id": track_id,
            "name": track.get("name", ""),
            "artist": track.get("artist", ""),
            "popularity_index": score,
        }

    def get_playlist(self, playlist_id: str) -> Dict:
        # use_legacy_default=False: existing intermediate spotify queries
        # have no flakiness_config and rely on the legacy default firing on
        # the 13th get_track_popularity_index call only.  Without this flag,
        # this newly-added override would *also* fire on the 13th get_playlist
        # call for them, breaking established behavior.
        if check_flaky(self, "get_playlist", use_legacy_default=False):
            raise SpotifyError(
                "INTERNAL_ERROR",
                "Upstream playlist service returned an internal error. Please try again.",
                playlist_id=playlist_id,
            )
        return super().get_playlist(playlist_id)

    def get_user_playlists(self) -> Dict:
        if check_flaky(self, "get_user_playlists", use_legacy_default=False):
            raise SpotifyError(
                "INTERNAL_ERROR",
                "Upstream playlist service returned an internal error. Please try again.",
            )
        return super().get_user_playlists()

    def get_track_details(self, track_id: str) -> Dict:
        if check_flaky(self, "get_track_details", use_legacy_default=False):
            raise SpotifyError(
                "INTERNAL_ERROR",
                "Upstream track-details service returned an internal error. Please try again.",
                track_id=track_id,
            )
        return super().get_track_details(track_id)
