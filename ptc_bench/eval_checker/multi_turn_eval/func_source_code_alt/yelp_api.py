"""Yelp variant with popularity scores and configurable transient failures.

Only get_business_popularity_index uses the legacy 13th-call failure default.
"""

from typing import Any, Dict

from ..func_source_code.yelp_api import YelpAPI as _BaseYelpAPI
from ..func_source_code.yelp_api import YelpError
from ._flaky import check_flaky


class YelpAPI(_BaseYelpAPI):
    def get_business_popularity_index(self, business_id: str) -> Dict[str, Any]:
        """
        Get the popularity index for a business this week.

        The popularity index is a score from roughly 10 to 100 computed from
        the business's rating and review volume; higher values indicate more
        popular businesses this week.

        Args:
            business_id (str): The business to look up.

        Returns:
            Dict[str, Any]: business_id, name, popularity_index.
        """
        biz = self._require_business(business_id)

        if check_flaky(self, "get_business_popularity_index"):
            raise YelpError(
                "INTERNAL_ERROR",
                "Upstream popularity service returned an internal error. Please try again.",
                suggested_action="Retry the call.",
                context={"business_id": business_id},
            )

        rating = float(biz.get("rating", 0) or 0)
        review_count = int(biz.get("review_count", 0) or 0)
        score = round(rating * 10 + min(review_count, 500) / 10, 1)
        return {
            "business_id": business_id,
            "name": biz["name"],
            "popularity_index": score,
        }

    def get_reviews_for_business(self, business_id: str, *args, **kwargs):
        if check_flaky(self, "get_reviews_for_business", use_legacy_default=False):
            raise YelpError(
                "INTERNAL_ERROR",
                "Upstream review service returned an internal error. Please try again.",
                suggested_action="Retry the call.",
                context={"business_id": business_id},
            )
        return super().get_reviews_for_business(business_id, *args, **kwargs)

    def search_businesses(self, *args, **kwargs):
        if check_flaky(self, "search_businesses", use_legacy_default=False):
            raise YelpError(
                "INTERNAL_ERROR",
                "Upstream search service returned an internal error. Please try again.",
                suggested_action="Retry the call.",
                context={"args": args, "kwargs": kwargs},
            )
        return super().search_businesses(*args, **kwargs)
