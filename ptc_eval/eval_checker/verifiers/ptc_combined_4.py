"""Verifier for ptc_combined_4.

Office team outing coordinator: find Team Outing event, pick venue with most
reviews, update event, Uber estimates, Venmo requests, Spotify playlist, DoorDash.
Dynamically derives expected values from initial_config.
"""

from __future__ import annotations

from typing import Any

from .core import VerificationResult, score_from_checks


def verify(
    final_answer: str | None,
    final_state: dict | None,
    initial_config: dict,
    ground_truth: dict,
) -> VerificationResult:
    try:
        checks: dict[str, bool] = {}

        # Discover Team Outing event and attendees from initial_config
        cal_cfg = initial_config.get("PersonalCalendarAPI", {})
        team_outing_id = None
        attendees = set()
        for eid, evt in cal_cfg.get("events", {}).items():
            if "team outing" in evt.get("title", "").lower():
                team_outing_id = eid
                attendees = set(evt.get("attendees", []))
                break

        # Find venue with most reviews from Yelp
        yelp_cfg = initial_config.get("YelpAPI", {})
        businesses = yelp_cfg.get("businesses", {})
        team_biz = [(bid, b) for bid, b in businesses.items()
                     if any(kw in " ".join(b.get("categories", [])).lower()
                            for kw in ["team", "building", "activity", "entertainment"])]
        if not team_biz:
            team_biz = list(businesses.items())
        best_venue = max(team_biz, key=lambda x: x[1].get("review_count", 0))[1] if team_biz else {}
        venue_name = best_venue.get("name", "").lower()

        # --- 1. Calendar event updated with venue --------------------------------
        cal = (final_state or {}).get("PersonalCalendarAPI", {})
        cal_events = cal.get("events", {})

        team_event = cal_events.get(team_outing_id, {}) if team_outing_id else {}
        if team_event:
            location = team_event.get("location", "").lower()
            checks["event_location_updated"] = (
                venue_name in location or
                best_venue.get("address", "").lower() in location or
                len(location) > 5  # at least some location was set
            )
        else:
            checks["event_location_updated"] = False

        # --- 2. Venmo requests sent to attendees --------------------------------
        venmo = (final_state or {}).get("VenmoAPI", {})
        txns = venmo.get("transactions", [])
        initial_txns = initial_config.get("VenmoAPI", {}).get("transactions", [])
        initial_ids = {t.get("id") for t in initial_txns}
        new_txns = [t for t in txns if t.get("id") not in initial_ids]

        venmo_receivers = {t.get("receiver") for t in new_txns}
        covered = attendees & venmo_receivers
        checks["venmo_requests_sent"] = len(covered) >= max(1, len(attendees) - 1)

        checks["venmo_amounts_positive"] = all(
            t.get("amount", 0) > 0 for t in new_txns
        ) if new_txns else False

        # --- 3. Spotify playlist created ----------------------------------------
        spotify = (final_state or {}).get("SpotifyAPI", {})
        all_pls = spotify.get("playlists", {})
        initial_pls = set(initial_config.get("SpotifyAPI", {}).get("playlists", {}).keys())
        new_pls = {pid: pl for pid, pl in all_pls.items() if pid not in initial_pls}

        team_pls = [pl for pl in new_pls.values()
                     if "team" in pl.get("name", "").lower()
                     or "outing" in pl.get("name", "").lower()
                     or "vibe" in pl.get("name", "").lower()]
        checks["spotify_playlist_created"] = len(team_pls) > 0

        if team_pls:
            track_count = len(team_pls[0].get("tracks", []))
            checks["spotify_has_tracks"] = track_count >= 5
        else:
            checks["spotify_has_tracks"] = False

        # Check tracks are from a relevant playlist in initial config
        if team_pls:
            pl_tracks = set(team_pls[0].get("tracks", []))
            source_tracks = set()
            for pid, pl in initial_config.get("SpotifyAPI", {}).get("playlists", {}).items():
                name = pl.get("name", "").lower()
                if "team" in name or "building" in name or "energy" in name:
                    source_tracks.update(pl.get("tracks", []))
                    break
            if source_tracks:
                overlap = pl_tracks & source_tracks
                checks["spotify_correct_tracks"] = len(overlap) >= min(5, len(source_tracks))
            else:
                checks["spotify_correct_tracks"] = len(pl_tracks) > 0
        else:
            checks["spotify_correct_tracks"] = False

        # --- 4. Answer mentions venue -------------------------------------------
        ans = (final_answer or "").lower()
        checks["mentions_venue"] = venue_name in ans if venue_name else len(ans) > 50

        passed, score = score_from_checks(checks)
        return VerificationResult(
            passed=passed, score=score, checks=checks,
            details={
                "team_outing_id": team_outing_id,
                "attendees": sorted(attendees),
                "expected_venue": best_venue.get("name", ""),
            },
        )
    except (KeyError, TypeError, IndexError) as e:
        return VerificationResult(passed=False, score=0.0, errors=[f"Verifier error: {e}"])
