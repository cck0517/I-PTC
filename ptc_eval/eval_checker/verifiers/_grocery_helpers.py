"""Shared helpers for ptc_mt_grocery_* verifiers.

All functions take the GroceryDeliveryAPI sub-config
(initial_config["GroceryDeliveryAPI"]) unless otherwise noted.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable


def stores(cfg: dict) -> dict[str, dict]:
    return cfg.get("stores", {})


def shopping_list(cfg: dict) -> list[str]:
    return cfg.get("shopping_list", [])


def gift_cards(cfg: dict) -> dict[str, float]:
    return dict(cfg.get("gift_cards", {}))


def store_carries(store: dict, item: str) -> bool:
    inv = store.get("inventory", {})
    if item not in inv:
        return False
    return inv[item].get("in_stock", True)


def item_to_carriers(cfg: dict) -> dict[str, list[tuple[str, str, float]]]:
    """For each shopping list item, list (store_id, store_name, unit_price)
    of every in-stock carrier.  Sorted by (unit_price asc, store_name asc)."""
    out: dict[str, list[tuple[str, str, float]]] = defaultdict(list)
    items = set(shopping_list(cfg))
    for sid, s in stores(cfg).items():
        inv = s.get("inventory", {})
        for item, d in inv.items():
            if item not in items:
                continue
            if not d.get("in_stock", True):
                continue
            out[item].append((sid, s.get("name", sid), float(d["price"])))
    for item in out:
        out[item].sort(key=lambda r: (r[2], r[1]))
    return dict(out)


def store_cost(store: dict, items: Iterable[str]) -> dict | None:
    """Compute itemized cost for a store buying `items`.  Returns None if any
    item is not carried in-stock at this store.  Matches the formula used by
    the existing single-turn ptc_grocery_single_vs_split_0 verifier:
        item_total = sum(unit_price)
        service_fee = round(item_total * service_fee_rate, 2)
        store_total = round(item_total + service_fee + delivery_fee, 2)
    """
    inv = store.get("inventory", {})
    item_total = 0.0
    items_list = list(items)
    for item in items_list:
        if item not in inv or not inv[item].get("in_stock", True):
            return None
        item_total += inv[item]["price"]
    item_total = round(item_total, 2)
    service = round(item_total * store.get("service_fee_rate", 0), 2)
    delivery = store.get("delivery_fee", 0)
    return {
        "items": items_list,
        "item_total": item_total,
        "service_fee": service,
        "delivery_fee": delivery,
        "subtotal": round(item_total + service + delivery, 2),
    }


def split_assignment(
    sa: dict, sb: dict, shopping: list[str]
) -> tuple[list[str], list[str]] | None:
    """Greedy split: each item goes to whichever of {sa, sb} is cheaper for it.
    Items only one carries go to that one.  Returns None if some item is
    carried by neither, or if the split is degenerate (all items on one side)."""
    inv_a, inv_b = sa.get("inventory", {}), sb.get("inventory", {})
    items_a: list[str] = []
    items_b: list[str] = []
    for item in shopping:
        a_ok = item in inv_a and inv_a[item].get("in_stock", True)
        b_ok = item in inv_b and inv_b[item].get("in_stock", True)
        if not a_ok and not b_ok:
            return None
        if a_ok and not b_ok:
            items_a.append(item)
        elif b_ok and not a_ok:
            items_b.append(item)
        else:
            if inv_a[item]["price"] <= inv_b[item]["price"]:
                items_a.append(item)
            else:
                items_b.append(item)
    if not items_a or not items_b:
        return None
    return items_a, items_b
