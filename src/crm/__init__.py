"""CRM store access."""
from functools import lru_cache

from config import settings

from .store import CRMStore


@lru_cache(maxsize=1)
def get_crm_store() -> CRMStore:
    """Return the shared CRM store."""
    return CRMStore(settings.crm_db_path)


__all__ = ["CRMStore", "get_crm_store"]
