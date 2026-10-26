"""Variant of YelpAPI used by the intermediate experiment.

Two transient-failure injection points:

  - ``get_business_popularity_index`` — the popularity-ranking method
    used by the original yelp_chinese family of intermediate queries.
    Falls back to legacy default-13 when no flakiness_config is set,
    preserving original behavior.
  - ``get_reviews_for_business`` — the per-business review getter used
    by the review-analysis queries (recent_drift, polarization,
    useful_top, harsh_reviewers, etc.).  Uses ``use_legacy_default=False``
    so legacy queries (which don't set flakiness_config) don't fire on
    this newly-added method.

The class is named ``YelpAPI`` so that tracebacks and repr strings
surfaced to the model do not broadcast that this is a variant; the
module lives at ``func_source_code_alt/yelp_api.py`` and is selected
via the ``YelpAPIAlt`` key in CLASS_FILE_PATH_MAPPING.
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
        # use_legacy_default=False: existing yelp_chinese-family queries
        # rely on the legacy default firing on get_business_popularity_index.
        # Without this flag the new override would also fire on the 13th
        # get_reviews_for_business call for them, breaking established
        # behavior.
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
