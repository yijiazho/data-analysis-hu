"""
Publisher for individual movie events to PubSub.
"""
import json
import logging
import time
from typing import Iterator, Tuple, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from google.cloud import pubsub_v1
from google.cloud.pubsub_v1.types import PubsubMessage

from pubsub_config import PubSubConfig
from event_schemas import MovieEventBuilder

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MovieEventPublisher:
    """Publisher for movie events to Google Cloud PubSub."""

    def __init__(self, config: PubSubConfig):
        """Initialize the publisher with configuration."""
        self.config = config
        self.publisher = pubsub_v1.PublisherClient()
        self.topic_path = self.publisher.topic_path(config.project_id, config.raw_events_topic)
        self.deadletter_topic_path = self.publisher.topic_path(config.project_id, config.deadletter_topic)

        # Configure batching settings
        batch_settings = pubsub_v1.types.BatchSettings(
            max_messages=config.batch_size,
            max_bytes=1024 * 1024 * 5,  # 5MB
            max_latency=config.max_batch_wait_time
        )

        # Configure publisher settings
        publisher_options = pubsub_v1.types.PublisherOptions(
            enable_message_ordering=False
        )

        self.publisher = pubsub_v1.PublisherClient(
            batch_settings=batch_settings,
            publisher_options=publisher_options
        )

        # Statistics
        self.stats = {
            'total_processed': 0,
            'total_published': 0,
            'total_failed': 0,
            'total_deadlettered': 0
        }

    def publish_movies_from_csv(self, csv_path: str) -> Dict[str, int]:
        """
        Publish movie events from CSV file.

        Args:
            csv_path: Path to the movies facts CSV file

        Returns:
            Dictionary with publishing statistics
        """
        logger.info(f"Starting to publish movies from {csv_path}")
        start_time = time.time()

        try:
            # Read CSV in chunks to manage memory
            chunk_iter = pd.read_csv(csv_path, chunksize=self.config.chunk_size)

            # Process chunks concurrently
            with ThreadPoolExecutor(max_workers=self.config.concurrent_workers) as executor:
                futures = []

                for chunk_num, chunk in enumerate(chunk_iter):
                    future = executor.submit(self._process_chunk, chunk, chunk_num)
                    futures.append(future)

                    # Rate limiting: don't submit too many chunks at once
                    if len(futures) >= self.config.concurrent_workers * 2:
                        # Wait for some to complete
                        for future in as_completed(futures[:len(futures)//2]):
                            chunk_stats = future.result()
                            self._update_stats(chunk_stats)
                        futures = futures[len(futures)//2:]

                # Wait for remaining futures
                for future in as_completed(futures):
                    chunk_stats = future.result()
                    self._update_stats(chunk_stats)

        except Exception as e:
            logger.error(f"Error processing CSV file: {e}")
            raise

        # Final statistics
        duration = time.time() - start_time
        self.stats['duration_seconds'] = duration
        self.stats['messages_per_second'] = self.stats['total_published'] / duration if duration > 0 else 0

        logger.info(f"Publishing completed. Stats: {self.stats}")
        return self.stats

    def _process_chunk(self, chunk: pd.DataFrame, chunk_num: int) -> Dict[str, int]:
        """Process a chunk of movie data."""
        chunk_stats = {
            'processed': 0,
            'published': 0,
            'failed': 0,
            'deadlettered': 0
        }

        logger.info(f"Processing chunk {chunk_num} with {len(chunk)} movies")

        # Batch for publishing
        messages_batch = []

        for idx, row in chunk.iterrows():
            try:
                chunk_stats['processed'] += 1

                # Create movie event
                event, attributes = MovieEventBuilder.create_movie_event(row)

                # Prepare PubSub message
                message_data = json.dumps(event).encode('utf-8')
                pubsub_message = PubsubMessage(
                    data=message_data,
                    attributes=attributes
                )

                messages_batch.append(pubsub_message)

                # Publish batch when it reaches the batch size
                if len(messages_batch) >= self.config.batch_size:
                    batch_stats = self._publish_batch(messages_batch)
                    chunk_stats['published'] += batch_stats['published']
                    chunk_stats['failed'] += batch_stats['failed']
                    chunk_stats['deadlettered'] += batch_stats['deadlettered']
                    messages_batch = []

            except Exception as e:
                logger.error(f"Error processing movie {row.get('id', 'unknown')}: {e}")
                chunk_stats['failed'] += 1

        # Publish remaining messages in batch
        if messages_batch:
            batch_stats = self._publish_batch(messages_batch)
            chunk_stats['published'] += batch_stats['published']
            chunk_stats['failed'] += batch_stats['failed']
            chunk_stats['deadlettered'] += batch_stats['deadlettered']

        logger.info(f"Chunk {chunk_num} completed: {chunk_stats}")
        return chunk_stats

    def _publish_batch(self, messages: list) -> Dict[str, int]:
        """Publish a batch of messages with retry logic."""
        batch_stats = {
            'published': 0,
            'failed': 0,
            'deadlettered': 0
        }

        for attempt in range(self.config.max_retries + 1):
            try:
                if attempt > 0:
                    # Exponential backoff
                    sleep_time = self.config.retry_backoff_base ** attempt
                    logger.info(f"Retrying batch publish after {sleep_time}s (attempt {attempt})")
                    time.sleep(sleep_time)

                # Publish messages individually for better error handling
                failed_messages = []

                for message in messages:
                    try:
                        future = self.publisher.publish(self.topic_path, message.data, **message.attributes)
                        # Wait for publish to complete
                        message_id = future.result(timeout=30)
                        batch_stats['published'] += 1
                        logger.debug(f"Published message: {message_id}")

                    except Exception as e:
                        logger.warning(f"Failed to publish message: {e}")
                        failed_messages.append(message)

                # If all succeeded, break
                if not failed_messages:
                    break

                # Retry only failed messages
                messages = failed_messages

            except Exception as e:
                logger.error(f"Batch publish attempt {attempt} failed: {e}")
                if attempt == self.config.max_retries:
                    # Send to deadletter topic
                    batch_stats['deadlettered'] += len(messages)
                    self._send_to_deadletter(messages, str(e))
                    break

        batch_stats['failed'] = len(messages) - batch_stats['published'] - batch_stats['deadlettered']
        return batch_stats

    def _send_to_deadletter(self, messages: list, error_reason: str):
        """Send failed messages to deadletter topic."""
        try:
            for message in messages:
                # Add error information to attributes
                deadletter_attributes = dict(message.attributes)
                deadletter_attributes.update({
                    'error_reason': error_reason,
                    'failed_at': str(int(time.time())),
                    'original_topic': self.config.raw_events_topic
                })

                future = self.publisher.publish(
                    self.deadletter_topic_path,
                    message.data,
                    **deadletter_attributes
                )
                future.result(timeout=30)

            logger.info(f"Sent {len(messages)} messages to deadletter topic")

        except Exception as e:
            logger.error(f"Failed to send messages to deadletter topic: {e}")

    def _update_stats(self, chunk_stats: Dict[str, int]):
        """Update overall statistics with chunk statistics."""
        self.stats['total_processed'] += chunk_stats['processed']
        self.stats['total_published'] += chunk_stats['published']
        self.stats['total_failed'] += chunk_stats['failed']
        self.stats['total_deadlettered'] += chunk_stats['deadlettered']

    def close(self):
        """Close the publisher and cleanup resources."""
        try:
            # Wait for all pending publishes to complete
            self.publisher.stop()
            logger.info("Publisher closed successfully")
        except Exception as e:
            logger.error(f"Error closing publisher: {e}")

def main():
    """Main function for testing the movie publisher."""
    config = PubSubConfig()
    publisher = MovieEventPublisher(config)

    try:
        # Test with the denormalized movie facts
        csv_path = "output_large_filtered/movies_facts_denorm.csv"
        stats = publisher.publish_movies_from_csv(csv_path)

        print(f"Publishing completed:")
        print(f"  Total processed: {stats['total_processed']}")
        print(f"  Total published: {stats['total_published']}")
        print(f"  Total failed: {stats['total_failed']}")
        print(f"  Total deadlettered: {stats['total_deadlettered']}")
        print(f"  Duration: {stats.get('duration_seconds', 0):.2f} seconds")
        print(f"  Rate: {stats.get('messages_per_second', 0):.2f} messages/second")

    finally:
        publisher.close()

if __name__ == "__main__":
    main()