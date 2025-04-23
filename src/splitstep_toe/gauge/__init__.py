"""
Gauge-sector helpers (placeholder).

Currently:
    • anomaly_cancelled()   – hard-coded True
"""
from .anomaly import anomaly_cancelled

__all__ = ["anomaly_cancelled"]
from .anomaly import anomaly_sums, anomaly_cancelled     # noqa: F401
__all__ = ["anomaly_sums", "anomaly_cancelled"]



