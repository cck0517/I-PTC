from copy import deepcopy
from typing import Dict, List, Optional, Union

DEFAULT_STATE = {
    "balance": 1000.0,
    "friends": [],
    "transactions": [],
    "pending_requests": [],
    "current_user": None,
    "request_counter": 0,
    "transaction_counter": 0,
}


class VenmoAPI:
    """
    A class representing the Venmo API for managing payments and transactions.

    This class provides methods for sending payments, requesting money,
    managing friends, and viewing transaction history.
    """

    def __init__(self):
        self.balance: float
        self.friends: List[str]
        self.transactions: List[Dict]
        self.pending_requests: List[Dict]
        self.current_user: Optional[str]
        self.request_counter: int
        self.transaction_counter: int
        self.username_to_id: Dict[str, str] = {}
        self._api_description = "This tool belongs to the VenmoAPI, which provides functionality for peer-to-peer payments, payment requests, and transaction management."

    def _load_scenario(self, scenario: dict, long_context=False) -> None:
        DEFAULT_STATE_COPY = deepcopy(DEFAULT_STATE)
        self.balance = scenario.get("balance", DEFAULT_STATE_COPY["balance"])
        self.friends = scenario.get("friends", DEFAULT_STATE_COPY["friends"])
        self.transactions = scenario.get("transactions", DEFAULT_STATE_COPY["transactions"])
        self.pending_requests = scenario.get("pending_requests", DEFAULT_STATE_COPY["pending_requests"])
        # Accept "user_id" as alias for "current_user" (shuo_ptc compat)
        self.current_user = scenario.get("current_user",
                                         scenario.get("user_id",
                                                      DEFAULT_STATE_COPY["current_user"]))
        # Load username_to_id if provided (shuo_ptc tasks include it)
        self.username_to_id = scenario.get("username_to_id", {})
        self.request_counter = scenario.get("request_counter", DEFAULT_STATE_COPY["request_counter"])
        self.transaction_counter = scenario.get("transaction_counter", DEFAULT_STATE_COPY["transaction_counter"])

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, VenmoAPI):
            return False
        for attr_name in vars(self):
            if attr_name.startswith("_"):
                continue
            if getattr(self, attr_name) != getattr(value, attr_name):
                return False
        return True

    def get_balance(self) -> Dict[str, float]:
        """
        Get the current Venmo balance.

        Returns:
            balance (float): Current balance in the Venmo account.
        """
        return {"balance": self.balance}

    def add_friend(self, username: str) -> Dict[str, bool]:
        """
        Add a friend to the Venmo friends list.

        Args:
            username (str): Username of the person to add as a friend.
        Returns:
            friend_status (bool): True if friend was added, False if already friends.
        """
        if username in self.friends:
            return {"friend_status": False}
        self.friends.append(username)
        return {"friend_status": True}

    def get_friends(self) -> Dict[str, List[str]]:
        """
        Get the list of Venmo friends.

        Returns:
            friends (List[str]): List of friend usernames.
        """
        return {"friends": self.friends}

    def send_payment(self, receiver: str, amount: float, note: str) -> Dict[str, Union[str, float, bool]]:
        """
        Send a payment to another Venmo user.

        Args:
            receiver (str): Username of the payment receiver.
            amount (float): Amount to send.
            note (str): Note or description for the payment.
        Returns:
            transaction_id (str): ID of the transaction.
            status (str): Status of the payment ('sent').
            amount (float): Amount sent.
            receiver (str): Receiver username.
            note (str): Payment note.
        """
        if amount <= 0:
            return {"error": "Amount must be positive."}
        if self.balance < amount:
            return {"error": f"Insufficient balance. Current balance: {self.balance}"}
        if receiver not in self.friends and receiver != self.current_user:
            return {"error": f"'{receiver}' is not in your friends list."}

        self.balance -= amount
        self.transaction_counter += 1
        txn_id = f"txn_{self.transaction_counter}"
        txn = {
            "id": txn_id,
            "sender": self.current_user,
            "receiver": receiver,
            "amount": amount,
            "note": note,
            "date": "2026-03-31",
            "type": "payment",
        }
        self.transactions.append(txn)
        return {
            "transaction_id": txn_id,
            "status": "sent",
            "amount": amount,
            "receiver": receiver,
            "note": note,
        }

    def request_payment(self, receiver: str, amount: float, note: str) -> Dict[str, Union[str, float]]:
        """
        Request a payment from another Venmo user.

        Args:
            receiver (str): Username of the person to request payment from.
            amount (float): Amount to request.
            note (str): Note or description for the request.
        Returns:
            request_id (str): ID of the payment request.
            status (str): Status of the request ('requested').
            amount (float): Amount requested.
            receiver (str): Receiver username.
            note (str): Request note.
        """
        if amount <= 0:
            return {"error": "Amount must be positive."}

        # Validate that the receiver is a known friend. Without this check,
        # passing a display name (e.g. "Alice") silently stores the wrong
        # identifier — the API returns success but verification fails.
        # Mirrors the validation in ``send_payment``.
        if receiver not in self.friends and receiver != self.current_user:
            return {
                "error": (
                    f"'{receiver}' is not in your friends list. "
                    f"Use venmo_api.get_friends() to see valid usernames."
                )
            }

        self.request_counter += 1
        req_id = f"req_{self.request_counter}"
        txn = {
            "id": req_id,
            "sender": self.current_user,
            "receiver": receiver,
            "amount": amount,
            "note": note,
            "date": "2026-03-31",
            "type": "request",
        }
        self.transactions.append(txn)
        return {
            "request_id": req_id,
            "status": "requested",
            "amount": amount,
            "receiver": receiver,
            "note": note,
        }

    def get_transaction_history(
        self, start_date: str = None, end_date: str = None
    ) -> Dict[str, List[Dict]]:
        """
        Get transaction history, optionally filtered by date range.

        Args:
            start_date (str): [Optional] Start date for filtering (YYYY-MM-DD).
            end_date (str): [Optional] End date for filtering (YYYY-MM-DD).
        Returns:
            transactions (List[Dict]): List of transaction records.
        """
        result = self.transactions
        if start_date:
            result = [t for t in result if t.get("date", "") >= start_date]
        if end_date:
            result = [t for t in result if t.get("date", "") <= end_date]
        return {"transactions": result}

    def get_pending_requests(self) -> Dict[str, List[Dict]]:
        """
        Get all pending payment requests.

        Returns:
            pending_requests (List[Dict]): List of pending payment requests.
        """
        return {"pending_requests": self.pending_requests}

    def accept_request(self, request_id: str) -> Dict[str, bool]:
        """
        Accept a pending payment request.

        Args:
            request_id (str): ID of the request to accept.
        Returns:
            accept_status (bool): True if the request was accepted successfully.
        """
        for i, req in enumerate(self.pending_requests):
            if req["id"] == request_id:
                if self.balance < req["amount"]:
                    return {"error": f"Insufficient balance to accept request. Balance: {self.balance}"}
                self.balance -= req["amount"]
                self.pending_requests.pop(i)
                self.transaction_counter += 1
                self.transactions.append({
                    "id": f"txn_{self.transaction_counter}",
                    "sender": self.current_user,
                    "receiver": req["from_user"],
                    "amount": req["amount"],
                    "note": req.get("note", ""),
                    "date": "2026-03-31",
                    "type": "payment",
                })
                return {"accept_status": True}
        return {"error": f"Request '{request_id}' not found."}

    def decline_request(self, request_id: str) -> Dict[str, bool]:
        """
        Decline a pending payment request.

        Args:
            request_id (str): ID of the request to decline.
        Returns:
            decline_status (bool): True if the request was declined successfully.
        """
        for i, req in enumerate(self.pending_requests):
            if req["id"] == request_id:
                self.pending_requests.pop(i)
                return {"decline_status": True}
        return {"error": f"Request '{request_id}' not found."}
