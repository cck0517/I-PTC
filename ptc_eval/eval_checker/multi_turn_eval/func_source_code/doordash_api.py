from copy import deepcopy
from typing import Dict, List, Optional, Union

DEFAULT_STATE = {
    "restaurants": {},
    "menus": {},
    "orders": [],
    "order_counter": 0,
    "current_location": "",
    "delivery_addresses": [],
}


class DoorDashAPI:
    """
    A class representing the DoorDash API for food delivery services.

    This class provides methods for searching restaurants, browsing menus,
    placing orders, tracking deliveries, and viewing order history.
    """

    def __init__(self):
        self.restaurants: Dict[str, Dict]
        self.menus: Dict[str, List[Dict]]
        self.orders: List[Dict]
        self.order_counter: int
        self.current_location: str
        self.delivery_addresses: List[str]
        self._api_description = "This tool belongs to the DoorDashAPI, which provides functionality for food delivery services including restaurant search, menu browsing, order placement, and delivery tracking."

    def _load_scenario(self, scenario: dict, long_context=False) -> None:
        DEFAULT_STATE_COPY = deepcopy(DEFAULT_STATE)
        self.restaurants = scenario.get("restaurants", DEFAULT_STATE_COPY["restaurants"])
        self.menus = scenario.get("menus", DEFAULT_STATE_COPY["menus"])
        self.orders = scenario.get("orders", DEFAULT_STATE_COPY["orders"])
        self.order_counter = scenario.get("order_counter", DEFAULT_STATE_COPY["order_counter"])
        self.current_location = scenario.get("current_location", DEFAULT_STATE_COPY["current_location"])
        self.delivery_addresses = scenario.get("delivery_addresses", DEFAULT_STATE_COPY["delivery_addresses"])

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, DoorDashAPI):
            return False
        for attr_name in vars(self):
            if attr_name.startswith("_"):
                continue
            if getattr(self, attr_name) != getattr(value, attr_name):
                return False
        return True

    def search_restaurants(
        self, query: str, location: str = "", sort_by: str = "rating"
    ) -> Dict[str, List[Dict]]:
        """
        Search for restaurants on DoorDash.

        Args:
            query (str): Search term (e.g., 'pizza', 'sushi', restaurant name).
            location (str): [Optional] Delivery location.
            sort_by (str): [Optional] Sort criteria ('rating', 'delivery_fee', 'estimated_delivery_time'). Default 'rating'.
        Returns:
            restaurants (List[Dict]): List of matching restaurant dictionaries.
        """
        query_lower = query.lower()
        matches = []
        for rest in self.restaurants.values():
            name_match = query_lower in rest["name"].lower()
            cuisine_match = query_lower in rest.get("cuisine", "").lower()
            if name_match or cuisine_match:
                matches.append(rest)

        reverse = sort_by == "rating"
        matches.sort(key=lambda x: x.get(sort_by, 0), reverse=reverse)
        return {"restaurants": matches}

    def get_restaurant_menu(self, restaurant_id: str) -> Dict[str, List[Dict]]:
        """
        Get the menu for a specific restaurant.

        Args:
            restaurant_id (str): ID of the restaurant.
        Returns:
            menu (List[Dict]): List of menu item dictionaries with item_id, name, price, category, is_popular.
        """
        if restaurant_id not in self.restaurants:
            return {"error": f"Restaurant '{restaurant_id}' not found."}
        menu = self.menus.get(restaurant_id, [])
        return {"menu": menu}

    def place_order(
        self, restaurant_id: str, items: List[str], delivery_address: str
    ) -> Dict[str, Union[str, float, int, List]]:
        """
        Place a delivery order from a restaurant.

        Args:
            restaurant_id (str): ID of the restaurant to order from.
            items (List[str]): List of item_ids to order.
            delivery_address (str): Address for delivery.
        Returns:
            order_id (str): Unique order identifier.
            restaurant_id (str): Restaurant ID.
            items (List[str]): Items ordered.
            total (float): Total cost including delivery fee.
            delivery_fee (float): Delivery fee.
            estimated_time (int): Estimated delivery time in minutes.
            status (str): Order status ('placed').
        """
        if restaurant_id not in self.restaurants:
            return {"error": f"Restaurant '{restaurant_id}' not found."}

        menu = self.menus.get(restaurant_id, [])
        menu_map = {item["item_id"]: item for item in menu}

        item_total = 0.0
        ordered_items = []
        for item_id in items:
            if item_id not in menu_map:
                return {"error": f"Menu item '{item_id}' not found."}
            item_total += menu_map[item_id]["price"]
            ordered_items.append(menu_map[item_id]["name"])

        rest = self.restaurants[restaurant_id]
        delivery_fee = rest.get("delivery_fee", 0.0)
        total = round(item_total + delivery_fee, 2)

        self.order_counter += 1
        order_id = f"order_{self.order_counter}"
        order = {
            "id": order_id,
            "restaurant_id": restaurant_id,
            "items": ordered_items,
            "total": total,
            "delivery_fee": delivery_fee,
            "status": "placed",
            "date": "2026-03-31",
            "rating": None,
        }
        self.orders.append(order)

        return {
            "order_id": order_id,
            "restaurant_id": restaurant_id,
            "items": ordered_items,
            "total": total,
            "delivery_fee": delivery_fee,
            "estimated_time": rest.get("estimated_delivery_time", 30),
            "status": "placed",
        }

    def get_order_status(self, order_id: str) -> Dict:
        """
        Get the status of a specific order.

        Args:
            order_id (str): ID of the order to check.
        Returns:
            Order dictionary with id, restaurant_id, items, total, status, date.
        """
        for order in self.orders:
            if order["id"] == order_id:
                return order
        return {"error": f"Order '{order_id}' not found."}

    def cancel_order(self, order_id: str) -> Dict[str, bool]:
        """
        Cancel an active order.

        Args:
            order_id (str): ID of the order to cancel.
        Returns:
            cancel_status (bool): True if the order was cancelled successfully.
        """
        for order in self.orders:
            if order["id"] == order_id:
                if order["status"] not in ("placed", "in_progress"):
                    return {"error": f"Cannot cancel order with status '{order['status']}'."}
                order["status"] = "cancelled"
                return {"cancel_status": True}
        return {"error": f"Order '{order_id}' not found."}

    def get_order_history(self, limit: int = 10) -> Dict[str, List[Dict]]:
        """
        Get order history.

        Args:
            limit (int): [Optional] Maximum number of orders to return. Default 10.
        Returns:
            orders (List[Dict]): List of order records.
        """
        return {"orders": self.orders[-limit:] if limit else self.orders}

    def rate_order(self, order_id: str, rating: int, comment: str = "") -> Dict[str, bool]:
        """
        Rate a delivered order.

        Args:
            order_id (str): ID of the order to rate.
            rating (int): Rating from 1 to 5.
            comment (str): [Optional] Comment about the order.
        Returns:
            rating_status (bool): True if the rating was submitted successfully.
        """
        for order in self.orders:
            if order["id"] == order_id:
                if order["status"] != "delivered":
                    return {"error": "Can only rate delivered orders."}
                if not 1 <= rating <= 5:
                    return {"error": "Rating must be between 1 and 5."}
                order["rating"] = rating
                return {"rating_status": True}
        return {"error": f"Order '{order_id}' not found."}

    def get_delivery_estimate(
        self, restaurant_id: str, delivery_address: str
    ) -> Dict[str, Union[int, float]]:
        """
        Get delivery time and fee estimate for a restaurant.

        Args:
            restaurant_id (str): ID of the restaurant.
            delivery_address (str): Address for delivery.
        Returns:
            estimated_time (int): Estimated delivery time in minutes.
            delivery_fee (float): Delivery fee.
        """
        if restaurant_id not in self.restaurants:
            return {"error": f"Restaurant '{restaurant_id}' not found."}
        rest = self.restaurants[restaurant_id]
        return {
            "estimated_time": rest.get("estimated_delivery_time", 30),
            "delivery_fee": rest.get("delivery_fee", 0.0),
        }
