"""Public exports for the CRM subpackage."""
from functools import lru_cache

from config import settings

from .store import CRMStore


@lru_cache(maxsize=1)
def get_crm_store() -> CRMStore:
    """Return the one shared CRMStore, built on first use.

    lru_cache replaces the module-level global each caller used to keep. Two
    modules kept their own, so two connections to the same file were opened
    for no reason.
    """
    return CRMStore(settings.crm_db_path)


__all__ = ["CRMStore", "get_crm_store"]
