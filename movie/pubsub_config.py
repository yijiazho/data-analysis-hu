"""
Configuration for PubSub movie events publisher.
"""
import os
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class PubSubConfig:
    """Configuration for PubSub operations."""
    project_id: str = os.getenv("GOOGLE_CLOUD_PROJECT", "your-project-id")

    # Topic names
    raw_events_topic: str = "movie-events-raw"
    analytics_topic: str = "movie-events-analytics"
    deadletter_topic: str = "movie-events-deadletter"

    # Publishing configuration
    batch_size: int = 1000
    max_batch_wait_time: int = 10  # seconds
    max_retries: int = 3
    retry_backoff_base: float = 2.0

    # Rate limiting
    messages_per_minute: int = 8000
    concurrent_workers: int = 5

    # Message retention
    message_retention_days: int = 7
    deadletter_retention_days: int = 30

    # Data processing
    chunk_size: int = 1000  # rows to process at once

    @property
    def raw_events_topic_path(self) -> str:
        return f"projects/{self.project_id}/topics/{self.raw_events_topic}"

    @property
    def analytics_topic_path(self) -> str:
        return f"projects/{self.project_id}/topics/{self.analytics_topic}"

    @property
    def deadletter_topic_path(self) -> str:
        return f"projects/{self.project_id}/topics/{self.deadletter_topic}"

# Rating tier mappings
RATING_TIERS = {
    (0.0, 4.0): "low",
    (4.0, 7.0): "medium",
    (7.0, 10.1): "high"
}

# Revenue tier mappings (in USD)
REVENUE_TIERS = {
    (0, 1_000_000): "indie",
    (1_000_000, 100_000_000): "mid",
    (100_000_000, float('inf')): "blockbuster"
}

# Genre mappings for attributes
GENRE_MAPPING = {
    "Science Fiction": "sci-fi",
    "TV Movie": "tv-movie"
}

def get_rating_tier(rating: float) -> str:
    """Get rating tier based on vote average."""
    if rating is None or rating != rating:  # Check for NaN
        return "unknown"

    for (min_val, max_val), tier in RATING_TIERS.items():
        if min_val <= rating < max_val:
            return tier
    return "unknown"

def get_revenue_tier(revenue: float) -> str:
    """Get revenue tier based on revenue amount."""
    if revenue is None or revenue != revenue:  # Check for NaN
        return "unknown"

    for (min_val, max_val), tier in REVENUE_TIERS.items():
        if min_val <= revenue < max_val:
            return tier
    return "unknown"

def normalize_genre(genre: str) -> str:
    """Normalize genre name for attributes."""
    return GENRE_MAPPING.get(genre, genre.lower().replace(" ", "-"))