"""Data layer for market data integration.

This module provides:
- PolygonClient: Async client for Polygon.io API
- DataSourceRouter: Fallback chain orchestration
- Data models: Quote, Bar, CompanyInfo, NewsArticle
"""

from src.data.models import Bar, CompanyInfo, NewsArticle, Quote
from src.data.polygon import PolygonClient
from src.data.router import DataSourceRouter

__all__ = [
    "Bar",
    "CompanyInfo",
    "DataSourceRouter",
    "NewsArticle",
    "PolygonClient",
    "Quote",
]
