# Movie Data to PubSub Events

This implementation converts denormalized movie data into Google Cloud PubSub events following the design specified in `PUBSUB_DESIGN.md`.

## Quick Start

### 1. Setup Environment

```bash
# Install dependencies
pip install -r requirements.txt

# Set your Google Cloud project
export GOOGLE_CLOUD_PROJECT="your-project-id"

# Authenticate with Google Cloud
gcloud auth application-default login
```

### 2. Create PubSub Infrastructure

```bash
# Create topics and example subscriptions
python setup_pubsub_topics.py --setup-all

# Or create step by step
python setup_pubsub_topics.py --create-topics
python setup_pubsub_topics.py --create-subscriptions
```

### 3. Publish Movie Events

```bash
# Publish all events (movies + analytics)
python publish_movie_events.py

# Publish only movies
python publish_movie_events.py --skip-analytics

# Publish only analytics
python publish_movie_events.py --skip-movies

# Dry run to validate inputs
python publish_movie_events.py --dry-run
```

## Architecture Overview

### PubSub Topics

- **`movie-events-raw`** - Individual movie events (~1.27M events)
- **`movie-events-analytics`** - Pre-computed aggregation events (~500 events)
- **`movie-events-deadletter`** - Failed/invalid events

### Event Types

#### Movie Event Schema
```json
{
  "specversion": "1.0",
  "type": "movie.created",
  "source": "movie-denormalizer",
  "id": "movie-27205-1632234567",
  "time": "2024-09-20T13:19:00Z",
  "data": {
    "movieId": "27205",
    "title": "Inception",
    "ratings": { "average": 8.364, "count": 34495, "bin": "8.00–8.99" },
    "financial": { "revenue": 825532764, "budget": 160000000, "revenueBin": "500,000,000–999,999,999" },
    "technical": { "runtime": 148, "runtimeBin": "140–199", "adult": false }
  }
}
```

#### Analytics Event Schema
```json
{
  "specversion": "1.0",
  "type": "analytics.aggregation",
  "source": "movie-denormalizer",
  "data": {
    "aggregationType": "by_genre",
    "dimension": "Drama",
    "metrics": { "movieCount": 242114, "percentage": 19.02 }
  }
}
```

## Components

### Core Files

- **`pubsub_config.py`** - Configuration and utility functions
- **`event_schemas.py`** - CloudEvent schemas and builders
- **`movie_publisher.py`** - Publisher for individual movie events
- **`analytics_publisher.py`** - Publisher for analytics aggregation events
- **`batch_processor.py`** - Batch processing with error handling and rate limiting
- **`publish_movie_events.py`** - Main orchestrator script

### Utility Scripts

- **`setup_pubsub_topics.py`** - Create/manage PubSub topics and subscriptions

## Usage Examples

### Basic Publishing

```bash
# Default: publish everything from output_large_filtered/
python publish_movie_events.py

# Custom paths
python publish_movie_events.py \
  --movies-csv /path/to/movies.csv \
  --analytics-dir /path/to/analytics/

# Custom configuration
python publish_movie_events.py \
  --project-id my-project \
  --batch-size 500 \
  --max-workers 3 \
  --rate-limit 5000
```

### Advanced Options

```bash
# Export statistics
python publish_movie_events.py --export-stats publishing_stats.json

# Verbose logging
python publish_movie_events.py --verbose

# Validate without publishing
python publish_movie_events.py --dry-run
```

### Topic Management

```bash
# List existing topics
python setup_pubsub_topics.py --list-topics

# List subscriptions
python setup_pubsub_topics.py --list-subscriptions

# Delete topics (careful!)
python setup_pubsub_topics.py --delete-topics --confirm
```

## Message Attributes

Messages include attributes for filtering and routing:

### Movie Events
- `eventType`: `movie.created`
- `language`: Original language (e.g., `en`, `fr`)
- `decade`: Release decade (e.g., `2010s`)
- `ratingTier`: `low`, `medium`, `high`
- `revenueTier`: `indie`, `mid`, `blockbuster`
- `adult`: `true`, `false`

### Analytics Events
- `eventType`: `analytics.aggregation`
- `aggregationType`: `by_genre`, `by_region`, etc.
- `dimension`: Specific value (e.g., `Drama`, `United States`)
- `dataSource`: Source CSV filename

## Subscription Examples

### Filter High-Rated Movies
```bash
gcloud pubsub subscriptions create high-rated-movies \
  --topic=movie-events-raw \
  --message-filter='attributes.ratingTier="high"'
```

### Filter Action Movies
```bash
gcloud pubsub subscriptions create action-movies \
  --topic=movie-events-raw \
  --message-filter='attributes.genre:action'
```

### Genre Analytics Only
```bash
gcloud pubsub subscriptions create genre-analytics \
  --topic=movie-events-analytics \
  --message-filter='attributes.aggregationType="by_genre"'
```

## Monitoring and Statistics

### Real-time Status

```python
from publish_movie_events import MovieEventOrchestrator
from pubsub_config import PubSubConfig

orchestrator = MovieEventOrchestrator(PubSubConfig())
status = orchestrator.get_status()
print(status)
```

### Export Statistics

```bash
# During publishing
python publish_movie_events.py --export-stats results.json

# View statistics
cat results.json | jq '.global_stats'
```

## Performance Characteristics

### Throughput
- **Batch Size**: 1000 messages per batch
- **Rate Limit**: 8000 messages per minute (configurable)
- **Concurrency**: 5 workers (configurable)
- **Expected Duration**: ~15-20 minutes for full dataset

### Resource Usage
- **Memory**: ~200-500MB peak (chunked processing)
- **Network**: ~1.5GB total data transfer
- **PubSub Quota**: Well within default limits

## Error Handling

### Retry Logic
- **Max Retries**: 3 attempts per message
- **Backoff**: Exponential (2^attempt seconds)
- **Deadletter**: Failed messages sent to deadletter topic

### Monitoring Failed Events

```bash
# Subscribe to deadletter topic
gcloud pubsub subscriptions pull deadletter-monitor --auto-ack
```

### Recovery

```python
# Reprocess from deadletter topic
# Implementation depends on specific failure patterns
```

## Configuration

### Environment Variables

```bash
export GOOGLE_CLOUD_PROJECT="your-project-id"
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account.json"
```

### Configuration Override

```python
from pubsub_config import PubSubConfig

config = PubSubConfig()
config.project_id = "my-project"
config.batch_size = 500
config.max_retries = 5
```

## Troubleshooting

### Common Issues

1. **Authentication Error**
   ```bash
   gcloud auth application-default login
   ```

2. **Topic Not Found**
   ```bash
   python setup_pubsub_topics.py --create-topics
   ```

3. **Rate Limiting**
   - Reduce `--rate-limit` parameter
   - Increase `--max-workers` for better parallelization

4. **Memory Issues**
   - Reduce batch size with `--batch-size`
   - Process smaller CSV chunks

### Debug Logging

```bash
python publish_movie_events.py --verbose
```

### Validate Data

```bash
# Check CSV structure
head -n 5 output_large_filtered/movies_facts_denorm.csv

# Count rows
wc -l output_large_filtered/movies_facts_denorm.csv

# Validate before publishing
python publish_movie_events.py --dry-run
```

## Integration Examples

### Consuming Events

```python
from google.cloud import pubsub_v1

subscriber = pubsub_v1.SubscriberClient()
subscription_path = subscriber.subscription_path(project_id, subscription_id)

def callback(message):
    print(f"Received: {message.data}")
    message.ack()

streaming_pull_future = subscriber.subscribe(subscription_path, callback=callback)
```

### Real-time Analytics Pipeline

```python
# Dataflow job consuming movie events
# BigQuery integration for analytics
# ML model training with streaming data
```

## Cost Estimation

### PubSub Costs (approximate)
- **1.27M movie events**: ~$1.27 (at $1/million messages)
- **500 analytics events**: ~$0.001
- **Storage**: Minimal (7-day retention)
- **Total**: ~$1.30 for full dataset publish

### Optimization Tips
- Use batch publishing (implemented)
- Leverage message attributes for filtering
- Set appropriate retention periods
- Monitor and tune worker count