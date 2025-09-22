"""
Event schemas for movie PubSub events following CloudEvents specification.
"""
import json
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
import pandas as pd

from pubsub_config import get_rating_tier, get_revenue_tier, normalize_genre

@dataclass
class MovieRatings:
    """Movie rating information."""
    average: Optional[float] = None
    count: Optional[int] = None
    bin: Optional[str] = None

@dataclass
class MovieFinancial:
    """Movie financial information."""
    revenue: Optional[float] = None
    budget: Optional[float] = None
    revenue_bin: Optional[str] = None

@dataclass
class MovieTechnical:
    """Movie technical information."""
    runtime: Optional[int] = None
    runtime_bin: Optional[str] = None
    adult: Optional[bool] = None

@dataclass
class MovieData:
    """Core movie data payload."""
    movie_id: str
    title: str
    original_title: Optional[str] = None
    original_language: Optional[str] = None
    release_date: Optional[str] = None
    release_year: Optional[int] = None
    release_decade: Optional[int] = None
    ratings: Optional[MovieRatings] = None
    financial: Optional[MovieFinancial] = None
    technical: Optional[MovieTechnical] = None
    popularity: Optional[float] = None

@dataclass
class AnalyticsMetrics:
    """Analytics metrics information."""
    movie_count: int
    percentage: Optional[float] = None
    total_movies: Optional[int] = None

@dataclass
class AnalyticsMetadata:
    """Analytics metadata."""
    calculated_at: str
    data_version: str = "v1.0"

@dataclass
class AnalyticsData:
    """Analytics event data payload."""
    aggregation_type: str
    dimension: str
    metrics: AnalyticsMetrics
    metadata: AnalyticsMetadata

class MovieEventBuilder:
    """Builder for movie CloudEvents."""

    @staticmethod
    def create_movie_event(movie_row: pd.Series) -> Dict[str, Any]:
        """Create a CloudEvent for an individual movie."""
        event_time = datetime.now(timezone.utc).isoformat()
        movie_id = str(movie_row.get('id', ''))

        # Build movie data
        ratings = MovieRatings(
            average=_safe_float(movie_row.get('vote_average')),
            count=_safe_int(movie_row.get('vote_count')),
            bin=_safe_str(movie_row.get('rating_bin'))
        )

        financial = MovieFinancial(
            revenue=_safe_float(movie_row.get('revenue')),
            budget=_safe_float(movie_row.get('budget')),
            revenue_bin=_safe_str(movie_row.get('revenue_bin'))
        )

        technical = MovieTechnical(
            runtime=_safe_int(movie_row.get('runtime')),
            runtime_bin=_safe_str(movie_row.get('runtime_bin')),
            adult=_safe_bool(movie_row.get('adult'))
        )

        movie_data = MovieData(
            movie_id=movie_id,
            title=_safe_str(movie_row.get('title', '')),
            original_title=_safe_str(movie_row.get('original_title')),
            original_language=_safe_str(movie_row.get('original_language')),
            release_date=_safe_str(movie_row.get('release_date')),
            release_year=_safe_int(movie_row.get('release_year')),
            release_decade=_safe_int(movie_row.get('release_decade')),
            ratings=ratings,
            financial=financial,
            technical=technical,
            popularity=_safe_float(movie_row.get('popularity'))
        )

        # Build attributes
        attributes = MovieEventBuilder._build_movie_attributes(movie_row)

        # Create CloudEvent
        event = {
            "specversion": "1.0",
            "type": "movie.created",
            "source": "movie-denormalizer",
            "id": f"movie-{movie_id}-{int(datetime.now().timestamp())}",
            "time": event_time,
            "datacontenttype": "application/json",
            "data": _dataclass_to_dict(movie_data)
        }

        return event, attributes

    @staticmethod
    def _build_movie_attributes(movie_row: pd.Series) -> Dict[str, str]:
        """Build message attributes for filtering."""
        attributes = {
            "eventType": "movie.created",
            "source": "movie-denormalizer"
        }

        # Add language if available
        if pd.notna(movie_row.get('original_language')):
            attributes["language"] = str(movie_row['original_language']).lower()

        # Add decade
        if pd.notna(movie_row.get('release_decade')):
            decade = int(movie_row['release_decade'])
            attributes["decade"] = f"{decade}s"

        # Add rating tier
        vote_avg = movie_row.get('vote_average')
        if pd.notna(vote_avg):
            attributes["ratingTier"] = get_rating_tier(float(vote_avg))

        # Add revenue tier
        revenue = movie_row.get('revenue')
        if pd.notna(revenue):
            attributes["revenueTier"] = get_revenue_tier(float(revenue))

        # Add adult flag
        if pd.notna(movie_row.get('adult')):
            attributes["adult"] = str(movie_row['adult']).lower()

        return attributes

class AnalyticsEventBuilder:
    """Builder for analytics CloudEvents."""

    @staticmethod
    def create_analytics_event(
        aggregation_type: str,
        dimension: str,
        movie_count: int,
        data_source: str,
        total_movies: Optional[int] = None
    ) -> Dict[str, Any]:
        """Create a CloudEvent for analytics data."""
        event_time = datetime.now(timezone.utc).isoformat()

        # Calculate percentage if total provided
        percentage = None
        if total_movies and total_movies > 0:
            percentage = round((movie_count / total_movies) * 100, 2)

        metrics = AnalyticsMetrics(
            movie_count=movie_count,
            percentage=percentage,
            total_movies=total_movies
        )

        metadata = AnalyticsMetadata(
            calculated_at=event_time
        )

        analytics_data = AnalyticsData(
            aggregation_type=aggregation_type,
            dimension=dimension,
            metrics=metrics,
            metadata=metadata
        )

        # Build attributes
        attributes = {
            "eventType": "analytics.aggregation",
            "aggregationType": aggregation_type,
            "dimension": dimension,
            "dataSource": data_source
        }

        # Create CloudEvent
        event = {
            "specversion": "1.0",
            "type": "analytics.aggregation",
            "source": "movie-denormalizer",
            "id": f"analytics-{aggregation_type}-{dimension}-{int(datetime.now().timestamp())}",
            "time": event_time,
            "datacontenttype": "application/json",
            "data": _dataclass_to_dict(analytics_data)
        }

        return event, attributes

# Utility functions
def _safe_str(value) -> Optional[str]:
    """Safely convert value to string."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    return str(value).strip() if str(value).strip() else None

def _safe_int(value) -> Optional[int]:
    """Safely convert value to int."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return None

def _safe_float(value) -> Optional[float]:
    """Safely convert value to float."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None

def _safe_bool(value) -> Optional[bool]:
    """Safely convert value to bool."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, bool):
        return value
    str_val = str(value).lower().strip()
    return str_val in ('true', '1', 'yes', 'on')

def _dataclass_to_dict(obj) -> Dict[str, Any]:
    """Convert dataclass to dict, handling nested objects and None values."""
    if obj is None:
        return None

    result = {}
    for key, value in asdict(obj).items():
        if value is not None:
            result[key] = value

    return result