"""
Publisher for analytics aggregation events to PubSub.
"""
import json
import logging
import time
import os
from typing import Dict, Any, List, Tuple
import pandas as pd
from google.cloud import pubsub_v1
from google.cloud.pubsub_v1.types import PubsubMessage

from pubsub_config import PubSubConfig
from event_schemas import AnalyticsEventBuilder

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AnalyticsEventPublisher:
    """Publisher for analytics aggregation events to Google Cloud PubSub."""

    def __init__(self, config: PubSubConfig):
        """Initialize the analytics publisher with configuration."""
        self.config = config
        self.publisher = pubsub_v1.PublisherClient()
        self.topic_path = self.publisher.topic_path(config.project_id, config.analytics_topic)
        self.deadletter_topic_path = self.publisher.topic_path(config.project_id, config.deadletter_topic)

        # Analytics file mapping
        self.analytics_files = {
            'by_genre': 'movies_by_genre.csv',
            'by_region': 'movies_by_region.csv',
            'by_year': 'movies_by_year.csv',
            'by_decade': 'movies_by_decade.csv',
            'by_rating_bin': 'movies_by_rating_bin.csv',
            'by_revenue_bin': 'movies_by_revenue_bin.csv',
            'by_runtime_bin': 'movies_by_runtime_bin.csv',
            'by_language': 'movies_by_language.csv'
        }

        # Column name mappings for different aggregation types
        self.column_mappings = {
            'by_genre': ('genre', 'movie_count'),
            'by_region': ('country', 'movie_count'),
            'by_year': ('release_year', 'movie_count'),
            'by_decade': ('release_decade', 'movie_count'),
            'by_rating_bin': ('rating_bin', 'movie_count'),
            'by_revenue_bin': ('revenue_bin', 'movie_count'),
            'by_runtime_bin': ('runtime_bin', 'movie_count'),
            'by_language': ('original_language', 'movie_count')
        }

        # Statistics
        self.stats = {
            'total_files_processed': 0,
            'total_events_published': 0,
            'total_failed': 0,
            'files_processed': {}
        }

    def publish_analytics_from_directory(self, output_dir: str) -> Dict[str, Any]:
        """
        Publish analytics events from all aggregation CSV files in a directory.

        Args:
            output_dir: Directory containing the aggregation CSV files

        Returns:
            Dictionary with publishing statistics
        """
        logger.info(f"Starting to publish analytics from {output_dir}")
        start_time = time.time()

        # Calculate total movies for percentage calculations
        total_movies = self._get_total_movies_count(output_dir)
        logger.info(f"Total movies for percentage calculation: {total_movies}")

        for aggregation_type, filename in self.analytics_files.items():
            file_path = os.path.join(output_dir, filename)

            if not os.path.exists(file_path):
                logger.warning(f"Analytics file not found: {file_path}")
                continue

            try:
                file_stats = self._publish_analytics_file(
                    file_path, aggregation_type, total_movies
                )
                self.stats['files_processed'][aggregation_type] = file_stats
                self.stats['total_files_processed'] += 1

            except Exception as e:
                logger.error(f"Error processing analytics file {file_path}: {e}")
                self.stats['total_failed'] += 1

        # Calculate final statistics
        duration = time.time() - start_time
        self.stats['duration_seconds'] = duration
        self.stats['total_events_published'] = sum(
            file_stat.get('published', 0)
            for file_stat in self.stats['files_processed'].values()
        )

        logger.info(f"Analytics publishing completed. Stats: {self.stats}")
        return self.stats

    def _get_total_movies_count(self, output_dir: str) -> int:
        """Calculate total movies count for percentage calculations."""
        try:
            # Try to get from genre file (usually most complete)
            genre_file = os.path.join(output_dir, 'movies_by_genre.csv')
            if os.path.exists(genre_file):
                df = pd.read_csv(genre_file)
                return df['movie_count'].sum()

            # Fallback to year file
            year_file = os.path.join(output_dir, 'movies_by_year.csv')
            if os.path.exists(year_file):
                df = pd.read_csv(year_file)
                # Exclude NaN years from total count
                return df[df['release_year'].notna()]['movie_count'].sum()

            logger.warning("Could not determine total movies count")
            return 0

        except Exception as e:
            logger.error(f"Error calculating total movies: {e}")
            return 0

    def _publish_analytics_file(
        self, file_path: str, aggregation_type: str, total_movies: int
    ) -> Dict[str, int]:
        """Publish analytics events from a single CSV file."""
        logger.info(f"Processing analytics file: {file_path}")

        file_stats = {
            'processed': 0,
            'published': 0,
            'failed': 0,
            'skipped': 0
        }

        try:
            df = pd.read_csv(file_path)
            dimension_col, count_col = self.column_mappings[aggregation_type]

            # Validate required columns
            if dimension_col not in df.columns or count_col not in df.columns:
                raise ValueError(f"Required columns not found: {dimension_col}, {count_col}")

            # Process each row
            messages_batch = []

            for idx, row in df.iterrows():
                try:
                    file_stats['processed'] += 1

                    # Skip rows with null dimensions or zero counts
                    dimension = row[dimension_col]
                    count = row[count_col]

                    if pd.isna(dimension) or count <= 0:
                        file_stats['skipped'] += 1
                        logger.debug(f"Skipping row with null dimension or zero count: {dimension}")
                        continue

                    # Create analytics event
                    event, attributes = AnalyticsEventBuilder.create_analytics_event(
                        aggregation_type=aggregation_type,
                        dimension=str(dimension),
                        movie_count=int(count),
                        data_source=os.path.basename(file_path),
                        total_movies=total_movies
                    )

                    # Prepare PubSub message
                    message_data = json.dumps(event).encode('utf-8')
                    pubsub_message = PubsubMessage(
                        data=message_data,
                        attributes=attributes
                    )

                    messages_batch.append(pubsub_message)

                except Exception as e:
                    logger.error(f"Error processing analytics row {idx}: {e}")
                    file_stats['failed'] += 1

            # Publish all messages for this file
            if messages_batch:
                publish_stats = self._publish_messages_batch(messages_batch, aggregation_type)
                file_stats['published'] = publish_stats['published']
                file_stats['failed'] += publish_stats['failed']

            logger.info(f"File {file_path} processed: {file_stats}")

        except Exception as e:
            logger.error(f"Error reading analytics file {file_path}: {e}")
            file_stats['failed'] = file_stats['processed']

        return file_stats

    def _publish_messages_batch(self, messages: List[PubsubMessage], aggregation_type: str) -> Dict[str, int]:
        """Publish a batch of analytics messages."""
        publish_stats = {
            'published': 0,
            'failed': 0
        }

        for attempt in range(self.config.max_retries + 1):
            try:
                if attempt > 0:
                    sleep_time = self.config.retry_backoff_base ** attempt
                    logger.info(f"Retrying {aggregation_type} publish after {sleep_time}s (attempt {attempt})")
                    time.sleep(sleep_time)

                failed_messages = []

                for message in messages:
                    try:
                        future = self.publisher.publish(self.topic_path, message.data, **message.attributes)
                        message_id = future.result(timeout=30)
                        publish_stats['published'] += 1
                        logger.debug(f"Published analytics message: {message_id}")

                    except Exception as e:
                        logger.warning(f"Failed to publish analytics message: {e}")
                        failed_messages.append(message)

                # If all succeeded, break
                if not failed_messages:
                    break

                # Retry only failed messages
                messages = failed_messages

            except Exception as e:
                logger.error(f"Analytics batch publish attempt {attempt} failed: {e}")
                if attempt == self.config.max_retries:
                    # Send to deadletter topic
                    self._send_to_deadletter(messages, str(e), aggregation_type)
                    break

        publish_stats['failed'] = len(messages)
        return publish_stats

    def _send_to_deadletter(self, messages: List[PubsubMessage], error_reason: str, aggregation_type: str):
        """Send failed analytics messages to deadletter topic."""
        try:
            for message in messages:
                deadletter_attributes = dict(message.attributes)
                deadletter_attributes.update({
                    'error_reason': error_reason,
                    'failed_at': str(int(time.time())),
                    'original_topic': self.config.analytics_topic,
                    'aggregation_type': aggregation_type
                })

                future = self.publisher.publish(
                    self.deadletter_topic_path,
                    message.data,
                    **deadletter_attributes
                )
                future.result(timeout=30)

            logger.info(f"Sent {len(messages)} analytics messages to deadletter topic")

        except Exception as e:
            logger.error(f"Failed to send analytics messages to deadletter topic: {e}")

    def close(self):
        """Close the publisher and cleanup resources."""
        try:
            self.publisher.stop()
            logger.info("Analytics publisher closed successfully")
        except Exception as e:
            logger.error(f"Error closing analytics publisher: {e}")

def main():
    """Main function for testing the analytics publisher."""
    config = PubSubConfig()
    publisher = AnalyticsEventPublisher(config)

    try:
        # Test with the analytics files
        output_dir = "output_large_filtered"
        stats = publisher.publish_analytics_from_directory(output_dir)

        print(f"Analytics publishing completed:")
        print(f"  Total files processed: {stats['total_files_processed']}")
        print(f"  Total events published: {stats['total_events_published']}")
        print(f"  Total failed: {stats['total_failed']}")
        print(f"  Duration: {stats.get('duration_seconds', 0):.2f} seconds")

        print(f"\nPer-file statistics:")
        for file_type, file_stats in stats['files_processed'].items():
            print(f"  {file_type}: {file_stats}")

    finally:
        publisher.close()

if __name__ == "__main__":
    main()