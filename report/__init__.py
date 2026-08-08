"""Reporting: aggregate stored results into per-category metrics and Markdown."""

from report.metrics import CampaignMetrics, CategoryMetrics, compute_metrics
from report.render import render_markdown, render_table

__all__ = [
    "CampaignMetrics",
    "CategoryMetrics",
    "compute_metrics",
    "render_markdown",
    "render_table",
]
