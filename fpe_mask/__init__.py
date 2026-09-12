"""fpe_mask: structure-aware, format-preserving masking for agent data pipelines."""

from .ff1 import FF1, DomainTooSmallError, FF1Error
from .engine import MaskingEngine, MaskResult, generate_master_key
from .suffixes import HUNGARIAN_COMPANY_FORMS

__all__ = [
    "FF1",
    "DomainTooSmallError",
    "FF1Error",
    "MaskingEngine",
    "MaskResult",
    "generate_master_key",
    "HUNGARIAN_COMPANY_FORMS",
]
