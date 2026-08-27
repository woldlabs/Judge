"""
Judge: Joint Unconventional Data & Geophysical Examination

A multimodal anomaly detection framework for systematic identification
of statistically and physically anomalous events in observational data.
"""

__version__ = "0.2.0"
__author__ = "woldlabs"

from judge.core.session import AnalysisSession
from judge.core.models import AnomalyEvent, Modality, AnalysisResult

__all__ = [
    "AnalysisSession",
    "AnomalyEvent",
    "Modality",
    "AnalysisResult",
]
