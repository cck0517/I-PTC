"""Shared helpers for ptc_mt_spotify_*_0 verifiers."""
from __future__ import annotations

from collections import defaultdict


def user_playlists(cfg: dict) -> list[dict]:
    """Return the list of playlist dicts for the user_playlists ids."""
    pls = cfg.get("playlists", {})
    return [pls[pid] for pid in cfg.get("user_playlists", []) if pid in pls]


def track_id_set_per_playlist(cfg: dict) -> dict[str, set[str]]:
    """{playlist_id: set(track_ids)} for the user_playlists, dedup'd within."""
    out: dict[str, set[str]] = {}
    for pl in user_playlists(cfg):
        out[pl["id"]] = set(pl.get("tracks", []))
    return out


def track_to_playlists(cfg: dict) -> dict[str, set[str]]:
    """{track_id: set(playlist_ids)} for tracks appearing in any user_playlists."""
    out: dict[str, set[str]] = defaultdict(set)
    for pl in user_playlists(cfg):
        for tid in set(pl.get("tracks", [])):
            out[tid].add(pl["id"])
    return dict(out)


def artist_to_tracks(cfg: dict) -> dict[str, set[str]]:
    """{artist: set(track_ids)} restricted to tracks in user_playlists."""
    tracks = cfg.get("tracks", {})
    seen: set[str] = set()
    for pl in user_playlists(cfg):
        seen.update(pl.get("tracks", []))
    out: dict[str, set[str]] = defaultdict(set)
    for tid in seen:
        t = tracks.get(tid)
        if not t:
            continue
        out[t["artist"]].add(tid)
    return dict(out)


def artist_to_playlists(cfg: dict) -> dict[str, set[str]]:
    """{artist: set(playlist_ids)} based on tracks appearing in user_playlists."""
    tracks = cfg.get("tracks", {})
    out: dict[str, set[str]] = defaultdict(set)
    for pl in user_playlists(cfg):
        for tid in set(pl.get("tracks", [])):
            t = tracks.get(tid)
            if not t:
                continue
            out[t["artist"]].add(pl["id"])
    return dict(out)


def playlist_name(cfg: dict, pid: str) -> str:
    return cfg.get("playlists", {}).get(pid, {}).get("name", "")


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)
