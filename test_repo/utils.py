"""
Helper functions used across the project.
"""

def format_currency(amount: float) -> str:
    """Return amount as USD string."""
    return f"${amount:.2f}"

def log_transaction(transaction_id: str, status: str) -> None:
    """Print log (mock logging)."""
    print(f"[LOG] {transaction_id}: {status}")