"""Compatibility layer for evidence aggregation.

New code should import from ``evidence.global_aggregator``.
"""

from evidence.global_aggregator import GlobalEvidenceAggregator

__all__ = ["GlobalEvidenceAggregator"]
