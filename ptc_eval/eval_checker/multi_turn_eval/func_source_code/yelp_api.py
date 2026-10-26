"""
Yelp Dummy API (in-memory, deterministic, benchmark-friendly)

Design goals:
- No real network requests; pure function calls.
- Explicit in-memory state seeded via _load_scenario().
- Structured errors (error_code, message, suggested_action, context).
- Current-user perspective: no registration or account switching.
- The current user can browse businesses, read any review, and write/edit/delete
  only their own reviews. One review per business per user.
"""

from __future__ import annotations

import copy
import math
import random
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .server_patch_mixin import PatchableMixin


# ---------------------------------------------------------------------------
# Error model
# ---------------------------------------------------------------------------


class YelpError(Exception):
    def __init__(
        self,
        error_code: str,
        message: str,
        suggested_action: str = "",
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.error = {
            "error_code": error_code,
            "message": message,
            "suggested_action": suggested_action,
            "context": context or {},
        }

    def to_dict(self) -> Dict[str, Any]:
        return copy.deepcopy(self.error)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


DEFAULT_STATE = {
    "random_seed": 7890,
    "profile": {},
    "businesses": {},
    "reviews": {},
    "saved_places": {},
    "bookmarks": [],
    "events": [],
}


class YelpAPI(PatchableMixin):
    """
    In-memory dummy implementation of a Yelp-like restaurant review service.

    State variables:
    - profile: {name, email} — the current user's identity.
    - businesses: Dict of {business_id -> {business_id, name, category, address,
      rating, review_count, price_level, opening_hours, phone, website}}
    - reviews: Dict of {review_id -> {review_id, business_id,
      author{name, is_current_user}, rating, text, photos[], created_at,
      helpful_count}}
    - saved_places: Dict of {business_id -> {business_id, name}}

    Current-user perspective: no registration or account switching.
    """


    def __init__(self):
        self._id_counters = {"review": 0}
        self.profile: Dict[str, Any] = {}
        self.businesses: Dict[str, Dict[str, Any]] = {}
        self.reviews: Dict[str, Dict[str, Any]] = {}
        self.saved_places: Dict[str, Dict[str, Any]] = {}
        self.bookmarks: List[str] = []
        self.events: List[Dict[str, Any]] = []
        self._api_description = (
            "This tool belongs to the Yelp API, which provides functionality "
            "for searching businesses, reading and writing reviews, and "
            "managing saved/bookmarked businesses."
        )


    def _new_id(self, prefix: str) -> str:
        """Generate the next sequential ID for *prefix* (e.g. ``order_1``)."""
        self._id_counters[prefix] = self._id_counters.get(prefix, 0) + 1
        return f"{prefix}_{self._id_counters[prefix]}"

    def _load_scenario(
        self,
        scenario: Dict[str, Any],
        long_context: bool = False,
    ) -> None:
        """
        Load a scenario from the scenarios folder.
        Args:
            scenario (Dict[str, Any]): The scenario to load
        """
        DEFAULT_STATE_COPY = deepcopy(DEFAULT_STATE)
        self._random = random.Random(
            scenario.get("random_seed", DEFAULT_STATE_COPY["random_seed"])
        )
        self.profile = scenario.get("profile", DEFAULT_STATE_COPY["profile"])
        self.businesses = scenario.get("businesses", DEFAULT_STATE_COPY["businesses"])
        self.reviews = scenario.get("reviews", DEFAULT_STATE_COPY["reviews"])
        # Normalize legacy schema {business_id: [review, ...]} into the
        # canonical {review_id: review_dict} expected by the rest of the API.
        # Some task configs use the legacy nested-list shape; without this
        # normalization, every get_reviews_for_business call raises
        # AttributeError because `.get()` is called on a list.
        if isinstance(self.reviews, dict) and self.reviews:
            sample = next(iter(self.reviews.values()))
            if isinstance(sample, list):
                normalized: Dict[str, Dict[str, Any]] = {}
                for biz_id, reviews_list in self.reviews.items():
                    if not isinstance(reviews_list, list):
                        continue
                    for i, review in enumerate(reviews_list):
                        if not isinstance(review, dict):
                            continue
                        review = dict(review)  # shallow copy
                        review.setdefault("business_id", biz_id)
                        rid = review.get("review_id") or f"rev_{biz_id}_{i + 1}"
                        review.setdefault("review_id", rid)
                        normalized[rid] = review
                self.reviews = normalized
        self.saved_places = scenario.get("saved_places", DEFAULT_STATE_COPY["saved_places"])
        self.bookmarks = scenario.get("bookmarks", DEFAULT_STATE_COPY["bookmarks"])
        self.events = scenario.get("events", DEFAULT_STATE_COPY["events"])
        self.long_context = long_context

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, YelpAPI):
            return False

        for attr_name in vars(self):
            if attr_name.startswith("_"):
                continue
            model_attr = getattr(self, attr_name)
            ground_truth_attr = getattr(value, attr_name)

            if model_attr != ground_truth_attr:
                return False

        return True

    # -----------------------------------------------------------------------
    # Internal mechanics
    # -----------------------------------------------------------------------

    def _require_business(self, business_id: str) -> Dict[str, Any]:
        biz = self.businesses.get(business_id)
        if not biz:
            raise YelpError(
                "BUSINESS_NOT_FOUND",
                f"Business '{business_id}' not found.",
                suggested_action="Use search_businesses() to find valid business IDs.",
                context={"business_id": business_id},
            )
        return biz

    def _require_review(self, review_id: str) -> Dict[str, Any]:
        review = self.reviews.get(review_id)
        if not review:
            raise YelpError(
                "REVIEW_NOT_FOUND",
                f"Review '{review_id}' not found.",
                suggested_action="Use get_reviews_for_business() to find valid review IDs.",
                context={"review_id": review_id},
            )
        return review

    def _require_own_review(self, review_id: str) -> Dict[str, Any]:
        review = self._require_review(review_id)
        if not review.get("author", {}).get("is_current_user"):
            raise YelpError(
                "NOT_YOUR_REVIEW",
                "You can only modify your own reviews.",
                suggested_action="Use get_my_reviews() to find your review IDs.",
                context={"review_id": review_id},
            )
        return review

    def _recalculate_business_rating(self, business_id: str) -> None:
        """Recompute average rating and review_count for a business."""
        biz = self.businesses.get(business_id)
        if not biz:
            return
        ratings = [
            r.get("rating", 0)
            for r in self.reviews.values()
            if r.get("business_id") == business_id
        ]
        if ratings:
            biz["rating"] = round(sum(ratings) / len(ratings), 1)
            biz["review_count"] = len(ratings)
        else:
            biz["rating"] = 0
            biz["review_count"] = 0

    # -----------------------------------------------------------------------
    # Profile
    # -----------------------------------------------------------------------

    def get_yelp_profile(self) -> Dict[str, Any]:
        """
        Get the current user's Yelp profile.

        Returns:
            Dict[str, Any]:
                name (str): The user's display name.
                email (str): The user's email address.
        """
        return deepcopy(self.profile)

    # -----------------------------------------------------------------------
    # Business discovery
    # -----------------------------------------------------------------------

    def search_businesses(
        self,
        query: Optional[str] = None,
        location: Optional[str] = None,
        category: Optional[str] = None,
        price_level: Optional[str] = None,
        min_rating: Optional[float] = None,
        sort_by: str = "rating",
        max_results: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Search for businesses by name, location, category, or other criteria.

        Args:
            query (str, optional): Text to match against business name.
            location (str, optional): Text to match against business address.
            category (str, optional): Case-insensitive match against any tag
                in the business's ``categories`` list. Example tags include
                cuisines like "Chinese" or "Italian", restaurant types like
                "restaurant" or "sushi", cafes/bars like "Cafe" or "Bakery",
                retail like "Supermarket" or "grocery", and venues/activities
                like "venue" or "live music". Use the singular form
                (e.g. "restaurant", not "restaurants").
            price_level (str, optional): Exact price level match
                (e.g. "$", "$$", "$$$", "$$$$").
            min_rating (float, optional): Minimum average rating (1.0–5.0).
            sort_by (str): Sort order. One of "rating", "review_count", "name".
                Defaults to "rating".
            max_results (int): Maximum results to return. Defaults to 20.

        Returns:
            List[Dict[str, Any]]: Matching business objects.
        """
        if sort_by not in ("rating", "review_count", "name"):
            raise YelpError(
                "INVALID_SORT",
                f"sort_by '{sort_by}' is not valid.",
                suggested_action="Use 'rating', 'review_count', or 'name'.",
                context={"sort_by": sort_by},
            )

        results = []
        for biz in self.businesses.values():
            match = True
            if query:
                q = query.lower()
                name_match = q in biz.get("name", "").lower()
                # Also match against categories list (e.g. ["Italian", "restaurant"])
                cats = biz.get("categories", [])
                cat_match = any(q in c.lower() for c in cats) if isinstance(cats, list) else False
                if not name_match and not cat_match:
                    match = False
            if location and location.lower() not in biz.get("address", "").lower():
                match = False
            if category:
                # Category match: case-insensitive lookup in the `categories` list.
                cats = biz.get("categories", [])
                if not (isinstance(cats, list)
                        and any(category.lower() == c.lower() for c in cats)):
                    match = False
            if price_level is not None and biz.get("price_level") != price_level:
                match = False
            if min_rating is not None and biz.get("rating", 0) < min_rating:
                match = False
            if match:
                results.append(deepcopy(biz))

        if sort_by == "rating":
            results.sort(key=lambda x: x.get("rating", 0), reverse=True)
        elif sort_by == "review_count":
            results.sort(key=lambda x: x.get("review_count", 0), reverse=True)
        elif sort_by == "name":
            results.sort(key=lambda x: x.get("name", ""))
        return results[:max_results]

    def get_business(self, business_id: str) -> Dict[str, Any]:
        """
        Get the full details of a business.

        Args:
            business_id (str): The unique identifier of the business.

        Returns:
            Dict[str, Any]: Business object including business_id, name,
                category, address, rating, review_count, price_level,
                opening_hours, phone, website.
        """
        biz = self._require_business(business_id)
        return deepcopy(biz)

    # -----------------------------------------------------------------------
    # Reviews — reading
    # -----------------------------------------------------------------------

    def get_reviews_for_business(
        self,
        business_id: str,
        max_results: int = 20,
        sort_by: str = "newest",
    ) -> List[Dict[str, Any]]:
        """
        Get reviews for a specific business.

        Args:
            business_id (str): The business to get reviews for.
            max_results (int): Maximum reviews to return. Defaults to 20.
            sort_by (str): Sort order. One of "newest", "oldest", "highest",
                "lowest", "most_helpful". Defaults to "newest".

        Returns:
            List[Dict[str, Any]]: Review objects sorted as requested.
        """
        self._require_business(business_id)
        if sort_by not in ("newest", "oldest", "highest", "lowest", "most_helpful"):
            raise YelpError(
                "INVALID_SORT",
                f"sort_by '{sort_by}' is not valid.",
                suggested_action="Use 'newest', 'oldest', 'highest', 'lowest', or 'most_helpful'.",
                context={"sort_by": sort_by},
            )

        results = [
            deepcopy(r) for r in self.reviews.values()
            if r.get("business_id") == business_id
        ]

        if sort_by == "newest":
            results.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        elif sort_by == "oldest":
            results.sort(key=lambda x: x.get("created_at", ""))
        elif sort_by == "highest":
            results.sort(key=lambda x: x.get("rating", 0), reverse=True)
        elif sort_by == "lowest":
            results.sort(key=lambda x: x.get("rating", 0))
        elif sort_by == "most_helpful":
            results.sort(key=lambda x: x.get("helpful_count", 0), reverse=True)
        return results[:max_results]

    def get_review_details(self, review_id: str) -> Dict[str, Any]:
        """
        Get the full details of a specific review by ID.

        Args:
            review_id (str): The unique identifier of the review.

        Returns:
            Dict[str, Any]: Full review object.
        """
        review = self._require_review(review_id)
        return deepcopy(review)

    def get_my_yelp_reviews(self) -> List[Dict[str, Any]]:
        """
        Get all Yelp reviews written by the current user.

        Returns:
            List[Dict[str, Any]]: The current user's reviews, newest first.
        """
        results = [
            deepcopy(r) for r in self.reviews.values()
            if r.get("author", {}).get("is_current_user")
        ]
        results.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return results

    # -----------------------------------------------------------------------
    # Reviews — writing (current user only)
    # -----------------------------------------------------------------------

    def compose_review(
        self,
        business_id: str,
        rating: int,
        text: str,
        photos: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Write a review for a business. Each user can only post one review per
        business. Use update_review() to modify an existing review.

        Args:
            business_id (str): The business to review.
            rating (int): Star rating from 1 to 5.
            text (str): Review text content.
            photos (List[str], optional): Photo URLs or filenames to attach.

        Returns:
            Dict[str, Any]:
                review_id (str), business_id (str), status (str).
        """
        self._require_business(business_id)
        if not isinstance(rating, int) or not 1 <= rating <= 5:
            raise YelpError(
                "INVALID_RATING",
                "Rating must be an integer between 1 and 5.",
                suggested_action="Provide a rating of 1, 2, 3, 4, or 5.",
                context={"rating": rating},
            )

        for r in self.reviews.values():
            if (r.get("business_id") == business_id
                    and r.get("author", {}).get("is_current_user")):
                raise YelpError(
                    "ALREADY_REVIEWED",
                    "You have already reviewed this business.",
                    suggested_action="Use update_review() to modify your existing review.",
                    context={
                        "business_id": business_id,
                        "existing_review_id": r["review_id"],
                    },
                )

        review_id = self._new_id("review")
        self.reviews[review_id] = {
            "review_id": review_id,
            "business_id": business_id,
            "author": {
                "name": self.profile.get("name", ""),
                "is_current_user": True,
            },
            "rating": rating,
            "text": text,
            "photos": photos or [],
            "created_at": _utc_now_iso(),
            "helpful_count": 0,
        }
        self._recalculate_business_rating(business_id)
        return {
            "review_id": review_id,
            "business_id": business_id,
            "status": "created",
        }

    def update_review(
        self,
        review_id: str,
        rating: Optional[int] = None,
        text: Optional[str] = None,
        photos: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Update one of your own reviews. Only the current user's reviews can be
        modified.

        Args:
            review_id (str): The review to update.
            rating (int, optional): New star rating (1–5). None keeps current.
            text (str, optional): New review text. None keeps current.
            photos (List[str], optional): New photo list. None keeps current.

        Returns:
            Dict[str, Any]: The updated review object.
        """
        review = self._require_own_review(review_id)
        if rating is not None:
            if not isinstance(rating, int) or not 1 <= rating <= 5:
                raise YelpError(
                    "INVALID_RATING",
                    "Rating must be an integer between 1 and 5.",
                    suggested_action="Provide a rating of 1, 2, 3, 4, or 5.",
                    context={"rating": rating},
                )
            review["rating"] = rating
        if text is not None:
            review["text"] = text
        if photos is not None:
            review["photos"] = photos
        self._recalculate_business_rating(review["business_id"])
        return deepcopy(review)

    def remove_review(self, review_id: str) -> Dict[str, Any]:
        """
        Remove one of your own Yelp reviews. Only the current user's reviews
        can be removed.

        Args:
            review_id (str): The review to remove.

        Returns:
            Dict[str, Any]:
                review_id (str), status (str).
        """
        review = self._require_own_review(review_id)
        business_id = review["business_id"]
        del self.reviews[review_id]
        self._recalculate_business_rating(business_id)
        return {"review_id": review_id, "status": "deleted"}

    # -----------------------------------------------------------------------
    # Review interaction
    # -----------------------------------------------------------------------

    def mark_review_useful(self, review_id: str) -> Dict[str, Any]:
        """
        Mark a review as useful. You cannot mark your own review.

        Args:
            review_id (str): The review to mark as useful.

        Returns:
            Dict[str, Any]:
                review_id (str), helpful_count (int).
        """
        review = self._require_review(review_id)
        if review.get("author", {}).get("is_current_user"):
            raise YelpError(
                "CANNOT_REACT_OWN_REVIEW",
                "You cannot mark your own review as useful.",
                suggested_action="Mark another user's review instead.",
                context={"review_id": review_id},
            )
        review["helpful_count"] = review.get("helpful_count", 0) + 1
        return {
            "review_id": review_id,
            "helpful_count": review["helpful_count"],
        }

    # -----------------------------------------------------------------------
    # Saved businesses
    # -----------------------------------------------------------------------

    def save_business(self, business_id: str) -> Dict[str, Any]:
        """
        Save (bookmark) a business for later reference.

        Args:
            business_id (str): The business to save.

        Returns:
            Dict[str, Any]:
                business_id (str), name (str), status (str).
        """
        biz = self._require_business(business_id)
        if business_id in self.saved_places:
            raise YelpError(
                "ALREADY_SAVED",
                f"Business '{business_id}' is already saved.",
                suggested_action="Use unsave_business() to remove it first.",
                context={"business_id": business_id},
            )
        self.saved_places[business_id] = {
            "business_id": business_id,
            "name": biz.get("name", ""),
        }
        return {
            "business_id": business_id,
            "name": biz.get("name", ""),
            "status": "saved",
        }

    def unsave_business(self, business_id: str) -> Dict[str, Any]:
        """
        Remove a business from saved/bookmarked list.

        Args:
            business_id (str): The business to unsave.

        Returns:
            Dict[str, Any]:
                business_id (str), status (str).
        """
        if business_id not in self.saved_places:
            raise YelpError(
                "NOT_SAVED",
                f"Business '{business_id}' is not in your saved list.",
                suggested_action="Use save_business() to save it first.",
                context={"business_id": business_id},
            )
        del self.saved_places[business_id]
        return {"business_id": business_id, "status": "unsaved"}

    def list_saved_businesses(self) -> List[Dict[str, Any]]:
        """
        List all saved/bookmarked businesses.

        Returns:
            List[Dict[str, Any]]: Saved place objects with business_id and name.
        """
        return [deepcopy(s) for s in self.saved_places.values()]

    # -----------------------------------------------------------------------
    # Bookmarks (list-based, from shuo_ptc tasks)
    # -----------------------------------------------------------------------

    def bookmark_business(self, business_id: str) -> Dict[str, bool]:
        """
        Bookmark a business for later reference.

        Args:
            business_id (str): Yelp business ID to bookmark.

        Returns:
            Dict[str, bool]: bookmark_status True if bookmarked, False if already bookmarked.
        """
        if business_id not in self.businesses:
            return {"error": f"Business '{business_id}' not found."}
        if business_id in self.bookmarks:
            return {"bookmark_status": False}
        self.bookmarks.append(business_id)
        return {"bookmark_status": True}

    def remove_bookmark(self, business_id: str) -> Dict[str, bool]:
        """
        Remove a business from bookmarks.

        Args:
            business_id (str): Yelp business ID to remove from bookmarks.

        Returns:
            Dict[str, bool]: remove_status True if removed, False if not in bookmarks.
        """
        if business_id not in self.bookmarks:
            return {"remove_status": False}
        self.bookmarks.remove(business_id)
        return {"remove_status": True}

    def get_bookmarks(self) -> Dict[str, List[Dict]]:
        """
        Get all bookmarked businesses with their details.

        Returns:
            Dict with 'bookmarks' list of bookmarked business dictionaries.
        """
        result = [deepcopy(self.businesses[bid]) for bid in self.bookmarks if bid in self.businesses]
        return {"bookmarks": result}

    # -----------------------------------------------------------------------
    # Events
    # -----------------------------------------------------------------------

    def search_events(self, term: str, location: str = "") -> Dict[str, List[Dict]]:
        """
        Search for local events on Yelp.

        Args:
            term (str): Search term for events.
            location (str): Optional location to filter events.

        Returns:
            Dict with 'events' list of matching event dictionaries.
        """
        term_lower = term.lower()
        matches = []
        for event in self.events:
            name_match = term_lower in event.get("name", "").lower()
            desc_match = term_lower in event.get("description", "").lower()
            if name_match or desc_match:
                matches.append(deepcopy(event))
        return {"events": matches}

    # -----------------------------------------------------------------------
    # Location & distance
    # -----------------------------------------------------------------------

    @staticmethod
    def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Return great-circle distance in miles between two lat/lon points."""
        R = 3958.8  # Earth radius in miles
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = (math.sin(dlat / 2) ** 2
             + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2)
        return round(2 * R * math.asin(math.sqrt(a)), 2)

    def get_distance_to_business(self, business_id: str) -> Dict[str, Any]:
        """
        Get the distance in miles from the current user's location to a
        business.

        Args:
            business_id (str): The business to measure distance to.

        Returns:
            Dict[str, Any]:
                business_id (str), name (str), distance_miles (float).
        """
        biz = self._require_business(business_id)
        user_loc = self.profile.get("location")
        if not user_loc:
            raise YelpError(
                "LOCATION_NOT_SET",
                "User location is not available in the profile.",
                suggested_action="Ensure the profile includes a 'location' with latitude and longitude.",
            )
        for key in ("latitude", "longitude"):
            if key not in biz:
                raise YelpError(
                    "NO_COORDINATES",
                    f"Business '{business_id}' does not have location coordinates.",
                    context={"business_id": business_id},
                )
        dist = self._haversine(
            user_loc["latitude"], user_loc["longitude"],
            biz["latitude"], biz["longitude"],
        )
        return {
            "business_id": business_id,
            "name": biz["name"],
            "distance_miles": dist,
        }

    def get_distance_between_businesses(
        self, business_id_1: str, business_id_2: str
    ) -> Dict[str, Any]:
        """
        Get the distance in miles between two businesses.

        Args:
            business_id_1 (str): First business.
            business_id_2 (str): Second business.

        Returns:
            Dict[str, Any]:
                business_id_1 (str), name_1 (str),
                business_id_2 (str), name_2 (str),
                distance_miles (float).
        """
        biz1 = self._require_business(business_id_1)
        biz2 = self._require_business(business_id_2)
        for b in (biz1, biz2):
            for key in ("latitude", "longitude"):
                if key not in b:
                    raise YelpError(
                        "NO_COORDINATES",
                        f"Business '{b['business_id']}' does not have location coordinates.",
                        context={"business_id": b["business_id"]},
                    )
        dist = self._haversine(
            biz1["latitude"], biz1["longitude"],
            biz2["latitude"], biz2["longitude"],
        )
        return {
            "business_id_1": business_id_1,
            "name_1": biz1["name"],
            "business_id_2": business_id_2,
            "name_2": biz2["name"],
            "distance_miles": dist,
        }

    # -----------------------------------------------------------------------
    # Business status
    # -----------------------------------------------------------------------

    def check_business_open(self, business_id: str) -> Dict[str, Any]:
        """
        Check whether a business is currently open based on its
        opening_hours and the simulated current time stored in the user
        profile (``profile["current_time"]``).

        Args:
            business_id (str): The business to check.

        Returns:
            Dict[str, Any]:
                business_id (str), name (str), is_open (bool),
                current_day (str), hours_today (str).
        """
        biz = self._require_business(business_id)
        ct_str = self.profile.get("current_time")
        if not ct_str:
            raise YelpError(
                "TIME_NOT_SET",
                "Simulated current time is not set in the profile.",
                suggested_action="Ensure profile includes 'current_time' in ISO format.",
            )
        ct = datetime.fromisoformat(ct_str)
        day_name = ct.strftime("%A")
        hours = biz.get("opening_hours", {})
        today_hours = hours.get(day_name)

        if today_hours is None or today_hours == "Closed":
            return {
                "business_id": business_id,
                "name": biz["name"],
                "is_open": False,
                "current_day": day_name,
                "hours_today": "Closed",
            }

        open_t = datetime.strptime(today_hours["open"], "%H:%M").time()
        close_t = datetime.strptime(today_hours["close"], "%H:%M").time()
        cur_t = ct.time()
        is_open = open_t <= cur_t < close_t

        return {
            "business_id": business_id,
            "name": biz["name"],
            "is_open": is_open,
            "current_day": day_name,
            "hours_today": f"{today_hours['open']} - {today_hours['close']}",
        }

    def get_wait_time(self, business_id: str) -> Dict[str, Any]:
        """
        Get the current estimated wait time for a business.  The wait time
        is stored as ``current_wait_minutes`` on the business object.

        Args:
            business_id (str): The business to query.

        Returns:
            Dict[str, Any]:
                business_id (str), name (str), has_wait_info (bool),
                estimated_wait_minutes (int) — only present when
                has_wait_info is True.
        """
        biz = self._require_business(business_id)
        wait = biz.get("current_wait_minutes")
        if wait is None:
            return {
                "business_id": business_id,
                "name": biz["name"],
                "has_wait_info": False,
                "message": "Wait time information is not available for this business.",
            }
        return {
            "business_id": business_id,
            "name": biz["name"],
            "has_wait_info": True,
            "estimated_wait_minutes": wait,
        }

    # ------------------------------------------------------------------
    # Reservation management
    # ------------------------------------------------------------------

    def make_reservation(
        self,
        business_id: str,
        date: str,
        time: str,
        party_size: int,
        name: str,
    ) -> Dict[str, Any]:
        """
        Make a reservation at a business.

        Validates that the business exists, accepts reservations
        (``accepts_reservations`` is True), is open on the requested day,
        the requested time falls within opening hours, and the party size
        does not exceed ``max_party_size``.

        Successful reservations are recorded in the business state and
        cannot be duplicated (same business + date + name → error).

        Args:
            business_id (str): The business to reserve at.
            date (str): Reservation date in YYYY-MM-DD format.
            time (str): Reservation time in HH:MM format (24-hour).
            party_size (int): Number of guests.
            name (str): Name for the reservation.

        Returns:
            Dict[str, Any]:
                On success: status ("confirmed"), reservation_id, business_id,
                business_name, date, time, party_size, name.
                On failure: status ("error"), error_code, message,
                suggested_action.
        """
        biz = self.businesses.get(business_id)
        if biz is None:
            return {
                "status": "error",
                "error_code": "BUSINESS_NOT_FOUND",
                "message": f"Business '{business_id}' not found.",
                "suggested_action": "Use search_businesses() to find valid business IDs.",
            }

        if not biz.get("accepts_reservations", False):
            return {
                "status": "error",
                "error_code": "NO_RESERVATIONS",
                "message": f"'{biz['name']}' does not accept reservations.",
                "suggested_action": "Skip this business or contact them directly.",
            }

        # Validate party size
        max_ps = biz.get("max_party_size", 10)
        if party_size > max_ps:
            return {
                "status": "error",
                "error_code": "PARTY_TOO_LARGE",
                "message": (
                    f"Party size {party_size} exceeds the maximum of "
                    f"{max_ps} for '{biz['name']}'."
                ),
                "suggested_action": f"Reduce party_size to {max_ps} or fewer.",
            }

        if party_size < 1:
            return {
                "status": "error",
                "error_code": "INVALID_PARTY_SIZE",
                "message": "Party size must be at least 1.",
                "suggested_action": "Provide a party_size >= 1.",
            }

        # Validate date / day-of-week against opening hours
        try:
            from datetime import datetime as _dt
            res_date = _dt.strptime(date, "%Y-%m-%d")
        except ValueError:
            return {
                "status": "error",
                "error_code": "INVALID_DATE",
                "message": f"'{date}' is not a valid YYYY-MM-DD date.",
                "suggested_action": "Provide date in YYYY-MM-DD format.",
            }

        day_name = res_date.strftime("%A")
        hours = biz.get("opening_hours", {})
        day_hours = hours.get(day_name)
        if day_hours is None or day_hours.get("open") is None:
            return {
                "status": "error",
                "error_code": "CLOSED_ON_DAY",
                "message": f"'{biz['name']}' is closed on {day_name}s.",
                "suggested_action": (
                    f"Choose a different date. Open days: "
                    f"{', '.join(d for d, h in hours.items() if h.get('open'))}."
                ),
            }

        # Validate time within opening hours
        open_t = day_hours["open"]
        close_t = day_hours["close"]
        if time < open_t or time >= close_t:
            return {
                "status": "error",
                "error_code": "OUTSIDE_HOURS",
                "message": (
                    f"Requested time {time} is outside opening hours "
                    f"({open_t}–{close_t}) for '{biz['name']}' on {day_name}."
                ),
                "suggested_action": f"Choose a time between {open_t} and {close_t}.",
            }

        # Check for duplicate reservation
        reservations = biz.setdefault("_reservations", [])
        for r in reservations:
            if r["date"] == date and r["name"] == name:
                return {
                    "status": "error",
                    "error_code": "DUPLICATE_RESERVATION",
                    "message": (
                        f"A reservation for '{name}' on {date} already exists "
                        f"at '{biz['name']}'."
                    ),
                    "suggested_action": "This reservation was already made.",
                }

        # Success — record the reservation
        res_id = f"res_{business_id}_{len(reservations)+1}"
        reservation = {
            "reservation_id": res_id,
            "business_id": business_id,
            "business_name": biz["name"],
            "date": date,
            "time": time,
            "party_size": party_size,
            "name": name,
        }
        reservations.append(reservation)

        return {"status": "confirmed", **reservation}