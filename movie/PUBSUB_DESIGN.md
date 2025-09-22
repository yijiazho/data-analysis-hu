# Movie Data to GCP PubSub Events - Design Document

## Overview

This document outlines the design for converting denormalized movie data into Google Cloud PubSub events to enable real-time analytics, streaming data processing, and event-driven architectures.

## Current Data Structure

### Source Files
- **Primary Data**: `movies_facts_denorm.csv` (161MB, ~1.27M movie records)
- **Aggregated Data**: 8 summary files with pre-computed analytics
  - `movies_by_genre.csv` - Movie counts by genre
  - `movies_by_region.csv` - Movie counts by country
  - `movies_by_year.csv` - Movie counts by release year
  - `movies_by_decade.csv` - Movie counts by decade
  - `movies_by_rating_bin.csv` - Movie counts by rating ranges
  - `movies_by_revenue_bin.csv` - Movie counts by revenue ranges
  - `movies_by_runtime_bin.csv` - Movie counts by runtime ranges
  - `movies_by_language.csv` - Movie counts by language

### Data Schema
Each movie record contains:
- **Identity**: ID, title, original title
- **Metadata**: Language, release date, year, decade
- **Ratings**: Vote average, vote count, rating bin
- **Financial**: Revenue, budget, revenue bin
- **Technical**: Runtime, runtime bin, adult flag
- **Popularity**: Popularity score

## PubSub Architecture Design

### Topic Structure

```
movie-events-raw          # Individual movie data events
movie-events-analytics    # Pre-computed aggregation events
movie-events-deadletter   # Failed/invalid events
```

### Event Schemas

#### Individual Movie Event

**Topic**: `movie-events-raw`

```json
{
  "eventType": "movie.created|movie.updated",
  "eventTime": "2024-09-20T13:19:00Z",
  "source": "movie-denormalizer",
  "specVersion": "1.0",
  "id": "movie-27205-20240920131900",
  "data": {
    "movieId": "27205",
    "title": "Inception",
    "originalTitle": "Inception",
    "originalLanguage": "en",
    "releaseDate": "2010-07-15",
    "releaseYear": 2010,
    "releaseDecade": 2010,
    "ratings": {
      "average": 8.364,
      "count": 34495,
      "bin": "8.00–8.99"
    },
    "financial": {
      "revenue": 825532764,
      "budget": 160000000,
      "revenueBin": "500,000,000–999,999,999"
    },
    "technical": {
      "runtime": 148,
      "runtimeBin": "140–199",
      "adult": false
    },
    "popularity": 83.952
  },
  "attributes": {
    "genre": "action,sci-fi,adventure",
    "country": "us,uk",
    "language": "en",
    "decade": "2010s",
    "ratingTier": "high",
    "revenueTier": "blockbuster"
  }
}
```

#### Analytics Event

**Topic**: `movie-events-analytics`

```json
{
  "eventType": "analytics.aggregation",
  "eventTime": "2024-09-20T13:19:00Z",
  "source": "movie-denormalizer",
  "specVersion": "1.0",
  "id": "analytics-genre-drama-20240920131900",
  "data": {
    "aggregationType": "by_genre",
    "dimension": "Drama",
    "metrics": {
      "movieCount": 242114,
      "percentage": 19.02,
      "totalMovies": 1272533
    },
    "metadata": {
      "calculatedAt": "2024-09-20T13:19:00Z",
      "dataVersion": "v1.0"
    }
  },
  "attributes": {
    "aggregationType": "by_genre",
    "dimension": "Drama",
    "dataSource": "movies_by_genre.csv"
  }
}
```

### Message Attributes Strategy

#### For Individual Movies (`movie-events-raw`)
- `eventType`: `movie.created`, `movie.updated`
- `language`: Original language code (e.g., `en`, `fr`, `ja`)
- `decade`: Release decade (e.g., `2010s`, `1990s`)
- `genre`: Primary genres (comma-separated)
- `country`: Production countries (comma-separated)
- `ratingTier`: `low` (0-4), `medium` (4-7), `high` (7-10)
- `revenueTier`: `indie` (<1M), `mid` (1M-100M), `blockbuster` (>100M)
- `adult`: `true`, `false`

#### For Analytics (`movie-events-analytics`)
- `eventType`: `analytics.aggregation`
- `aggregationType`: `by_genre`, `by_region`, `by_year`, etc.
- `dimension`: Specific value (e.g., `Drama`, `United States`)
- `dataSource`: Source file name

## Implementation Strategy

### Phase 1: Infrastructure Setup

#### Topic Creation
```bash
# Create topics with appropriate configurations
gcloud pubsub topics create movie-events-raw \
  --message-retention-duration=7d \
  --message-storage-policy-allowed-regions=us-central1

gcloud pubsub topics create movie-events-analytics \
  --message-retention-duration=7d \
  --message-storage-policy-allowed-regions=us-central1

gcloud pubsub topics create movie-events-deadletter \
  --message-retention-duration=30d \
  --message-storage-policy-allowed-regions=us-central1
```

#### Subscription Examples
```bash
# Real-time analytics subscription
gcloud pubsub subscriptions create movie-analytics-sub \
  --topic=movie-events-raw \
  --message-filter='attributes.ratingTier="high"'

# Genre-specific processing
gcloud pubsub subscriptions create action-movies-sub \
  --topic=movie-events-raw \
  --message-filter='attributes.genre:action'

# Aggregation data consumer
gcloud pubsub subscriptions create analytics-dashboard-sub \
  --topic=movie-events-analytics
```

### Phase 2: Data Publishers

#### Core Components

1. **Movie Data Publisher** (`movie_publisher.py`)
   - Reads `movies_facts_denorm.csv` in chunks
   - Converts each row to movie event
   - Publishes with appropriate attributes
   - Handles retries and error logging

2. **Analytics Publisher** (`analytics_publisher.py`)
   - Processes aggregation CSV files
   - Creates analytics events for each data point
   - Publishes to analytics topic

3. **Batch Processor** (`batch_processor.py`)
   - Manages chunking and rate limiting
   - Handles PubSub quotas and backpressure
   - Implements exponential backoff for retries

#### Publishing Strategy

**Batching Configuration**:
- Batch size: 1000 messages per publish call
- Max batch wait time: 10 seconds
- Concurrent publishers: 5 workers
- Rate limit: 8000 messages/minute (within PubSub quotas)

**Error Handling**:
- Retry failed publishes up to 3 times
- Exponential backoff: 2^attempt seconds
- Send to deadletter topic after max retries
- Log all failures for investigation

### Phase 3: Processing Pipeline

#### Data Flow

```
CSV Files → Data Publishers → PubSub Topics → Subscribers → Processing Systems
```

1. **Extract**: Read denormalized CSV files
2. **Transform**: Convert to CloudEvent format with attributes
3. **Load**: Publish to appropriate PubSub topics
4. **Process**: Consumers process events for various use cases

#### Processing Options

**Real-time Processing**:
- Dataflow streaming jobs for complex analytics
- Cloud Functions for simple transformations
- Custom applications for specialized processing

**Batch Processing**:
- BigQuery subscriptions for data warehousing
- Cloud Storage exports for archival
- Dataproc jobs for ML pipeline integration

## Use Cases and Consumers

### Real-time Analytics
- **Dashboard Updates**: Live movie statistics and trends
- **Recommendation Engines**: Genre and rating-based suggestions
- **Content Monitoring**: Adult content filtering and compliance

### Data Integration
- **Data Warehouse**: Historical analysis in BigQuery
- **ML Pipelines**: Training data for recommendation models
- **API Backends**: Real-time movie data serving

### Business Intelligence
- **Trend Analysis**: Popular genres and revenue patterns
- **Market Research**: Regional preferences and performance
- **Performance Metrics**: Rating distributions and popularity trends

## Technical Specifications

### Scalability Considerations

**Message Volume**:
- Individual movies: ~1.27M events (one-time load)
- Analytics events: ~500 events (aggregations)
- Estimated total size: ~1.5GB of event data

**Throughput Requirements**:
- Initial load: 1.27M messages over ~2 hours = ~175 messages/second
- Ongoing updates: Depends on data refresh frequency
- PubSub limit: 10,000 messages/second/topic (well within capacity)

### Cost Optimization

**Message Design**:
- Individual events: ~1-2KB each
- Use attributes for filtering to reduce unnecessary consumption
- Compress large payloads if needed

**Retention Policies**:
- Raw events: 7 days (for reprocessing)
- Analytics: 7 days (for dashboard stability)
- Deadletter: 30 days (for debugging)

### Monitoring and Observability

**Key Metrics**:
- Message publish success/failure rates
- Topic throughput and backlog
- Subscriber acknowledgment rates
- Processing latency per consumer

**Alerting**:
- Failed publish rate > 5%
- Deadletter topic message count > 100
- Subscriber lag > 1 hour
- Topic backlog > 10,000 messages

**Logging**:
- All publish operations with success/failure status
- Schema validation errors
- Retry attempts and final outcomes
- Consumer processing times

## Security and Compliance

### Access Control
- Use IAM roles for topic and subscription access
- Separate service accounts for publishers and consumers
- Principle of least privilege for all operations

### Data Privacy
- Exclude PII from message attributes
- Consider encryption for sensitive data
- Implement audit logging for data access

### Schema Evolution
- Version all event schemas
- Maintain backward compatibility
- Use schema registry for validation

## Migration and Deployment

### Deployment Strategy
1. **Development**: Test with sample data subset
2. **Staging**: Full dataset with monitoring
3. **Production**: Gradual rollout with feature flags

### Rollback Plan
- Maintain original CSV files as backup
- Implement consumer pause/resume functionality
- Plan for schema rollback if needed

### Performance Testing
- Load test with full dataset
- Validate consumer processing capacity
- Test failure scenarios and recovery

## Future Enhancements

### Real-time Updates
- Implement change data capture for live movie updates
- Stream processing for real-time analytics
- Event sourcing for complete audit trails

### Advanced Analytics
- Machine learning model training pipelines
- Complex event processing for trend detection
- Cross-dataset correlation analysis

### Integration Expansion
- Additional data sources (reviews, box office, streaming)
- External API integrations
- Multi-cloud event distribution