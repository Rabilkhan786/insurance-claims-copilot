"""Public exports for the policy-data subpackage."""
from functools import lru_cache

from config import settings

from .store import PolicyDataStore


@lru_cache(maxsize=1)
def get_policy_store() -> PolicyDataStore:
    """Return the one shared PolicyDataStore, built on first use."""
    return PolicyDataStore(settings.crm_db_path)


__all__ = ["PolicyDataStore", "get_policy_store"]
