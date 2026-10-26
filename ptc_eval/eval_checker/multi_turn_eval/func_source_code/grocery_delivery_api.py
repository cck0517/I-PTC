"""
Grocery Delivery Dummy API (in-memory, deterministic, benchmark-friendly)

Simulates a grocery delivery platform where users can browse stores,
check inventories and prices, and look up delivery fees and gift-card
balances.  No real network requests.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List

from .server_patch_mixin import PatchableMixin

DEFAULT_STATE: Dict[str, Any] = {
    "shopping_list": [],
    "stores": {},
    "gift_cards": {},
}


class GroceryDeliveryAPI(PatchableMixin):
    """
    In-memory mock of a grocery delivery service.

    State variables:
    - shopping_list: List of item name strings the user wants to buy.
    - stores: Dict mapping store_id to store info (name, delivery_fee,
      service_fee_rate, inventory).
    - gift_cards: Dict mapping store_id to gift-card balance (float).
    """

    def __init__(self) -> None:
        self.shopping_list: List[str] = []
        self.stores: Dict[str, Dict[str, Any]] = {}
        self.gift_cards: Dict[str, float] = {}
        self._api_description = (
            "This tool belongs to the GroceryDeliveryAPI mock, which "
            "provides grocery store inventory lookup, delivery fee "
            "information, and gift-card balance queries."
        )

    def _load_scenario(
        self, scenario: Dict[str, Any], long_context: bool = False
    ) -> None:
        d = deepcopy(DEFAULT_STATE)
        self.shopping_list = scenario.get("shopping_list", d["shopping_list"])
        self.stores = scenario.get("stores", d["stores"])
        self.gift_cards = scenario.get("gift_cards", d["gift_cards"])

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, GroceryDeliveryAPI):
            return False
        for attr_name in vars(self):
            if attr_name.startswith("_"):
                continue
            if getattr(self, attr_name) != getattr(value, attr_name):
                return False
        return True

    # ------------------------------------------------------------------
    # Shopping list
    # ------------------------------------------------------------------

    def get_shopping_list(self) -> Dict[str, Any]:
        """
        Return the current user's shopping list.

        Returns:
            Dict[str, Any]:
                items (list[str]): List of item names the user needs to buy.
        """
        return {"items": list(self.shopping_list)}

    # ------------------------------------------------------------------
    # Stores
    # ------------------------------------------------------------------

    def list_stores(self) -> Dict[str, Any]:
        """
        List all available grocery stores on the platform.

        Returns:
            Dict[str, Any]:
                stores (list[dict]): Each dict has store_id, name,
                    delivery_fee (float), and service_fee_rate (float,
                    e.g. 0.05 means 5%).
        """
        result = []
        for sid, info in self.stores.items():
            result.append({
                "store_id": sid,
                "name": info["name"],
                "delivery_fee": info["delivery_fee"],
                "service_fee_rate": info.get("service_fee_rate", 0.0),
            })
        return {"stores": result}

    def get_store_inventory(self, store_id: str) -> Dict[str, Any]:
        """
        Get the inventory (items and prices) for a specific store.

        Args:
            store_id (str): The store identifier.

        Returns:
            Dict[str, Any]:
                store_id (str): The queried store ID.
                store_name (str): The store name.
                inventory (list[dict]): Each dict has item_name (str),
                    price (float), and in_stock (bool).
        """
        store = self.stores.get(store_id)
        if store is None:
            return {"error": f"Store not found: {store_id!r}"}
        inv = store.get("inventory", {})
        items = []
        for item_name, details in inv.items():
            items.append({
                "item_name": item_name,
                "price": details["price"],
                "in_stock": details.get("in_stock", True),
            })
        return {
            "store_id": store_id,
            "store_name": store["name"],
            "inventory": items,
        }

    # ------------------------------------------------------------------
    # Gift cards
    # ------------------------------------------------------------------

    def get_gift_card_balance(self, store_id: str) -> Dict[str, Any]:
        """
        Check the user's gift-card balance for a specific store.

        Args:
            store_id (str): The store identifier.

        Returns:
            Dict[str, Any]:
                store_id (str): The queried store ID.
                balance (float): Gift-card balance in dollars.
                    Returns 0.0 if no gift card exists for this store.
        """
        store = self.stores.get(store_id)
        if store is None:
            return {"error": f"Store not found: {store_id!r}"}
        balance = self.gift_cards.get(store_id, 0.0)
        return {"store_id": store_id, "balance": balance}
