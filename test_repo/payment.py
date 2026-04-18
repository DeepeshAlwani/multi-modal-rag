"""
Payment processing module.
"""

def validate_card(card_number: str, expiry: str) -> bool:
    """
    Simulate card validation.
    Returns True if card number length is 16 and expiry not empty.
    """
    if len(card_number) == 16 and expiry:
        return True
    return False

def process_payment(amount: float, card_number: str, expiry: str) -> dict:
    """
    Process a payment after card validation.
    """
    if not validate_card(card_number, expiry):
        return {"status": "failed", "reason": "Invalid card"}
    # Simulate payment gateway call
    return {"status": "success", "amount": amount, "transaction_id": "txn_123"}