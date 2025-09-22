#!/usr/bin/env python3
"""
Main orchestrator script for publishing movie data to Google Cloud PubSub.

This script coordinates the publishing of both individual movie events and
analytics aggregation events to PubSub topics.
"""
import argparse
import logging
import os
import sys
import time
from typing import Dict, Any

from pubsub_config import PubSubConfig
from movie_publisher import MovieEventPublisher
from analytics_publisher import AnalyticsEventPublisher
from batch_processor import BatchProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MovieEventOrchestrator:
    """Orchestrates the publishing of movie events to PubSub."""

    def __init__(self, config: PubSubConfig):
        """Initialize the orchestrator."""
        self.config = config
        self.batch_processor = BatchProcessor(
            max_workers=config.concurrent_workers,
            max_messages_per_minute=config.messages_per_minute
        )

    def publish_all_events(
        self,
        movies_csv_path: str,
        analytics_dir: str,
        publish_movies: bool = True,
        publish_analytics: bool = True
    ) -> Dict[str, Any]:
        """
        Publish all movie events and analytics.

        Args:
            movies_csv_path: Path to the movie facts CSV file
            analytics_dir: Directory containing analytics CSV files
            publish_movies: Whether to publish individual movie events
            publish_analytics: Whether to publish analytics events

        Returns:
            Dictionary with overall publishing statistics
        """
        logger.info("Starting movie events publishing orchestration")
        start_time = time.time()

        # Validate inputs
        if publish_movies and not os.path.exists(movies_csv_path):
            raise FileNotFoundError(f"Movies CSV file not found: {movies_csv_path}")

        if publish_analytics and not os.path.isdir(analytics_dir):
            raise FileNotFoundError(f"Analytics directory not found: {analytics_dir}")

        # Submit jobs to batch processor
        job_ids = []

        if publish_movies:
            logger.info(f"Submitting movie publishing job for {movies_csv_path}")
            job_id = self.batch_processor.submit_job(
                job_id="movie_events",
                job_function=self._publish_movies,
                movies_csv_path=movies_csv_path
            )
            job_ids.append(job_id)

        if publish_analytics:
            logger.info(f"Submitting analytics publishing job for {analytics_dir}")
            job_id = self.batch_processor.submit_job(
                job_id="analytics_events",
                job_function=self._publish_analytics,
                analytics_dir=analytics_dir
            )
            job_ids.append(job_id)

        # Wait for all jobs to complete
        logger.info(f"Waiting for {len(job_ids)} jobs to complete...")
        completed_jobs = self.batch_processor.wait_for_jobs()

        # Collect results
        results = {
            'total_duration': time.time() - start_time,
            'jobs_completed': len(completed_jobs),
            'global_stats': self.batch_processor.get_global_stats(),
            'job_details': {}
        }

        for job in completed_jobs:
            results['job_details'][job.job_id] = job.to_dict()

        logger.info(f"Publishing orchestration completed in {results['total_duration']:.2f} seconds")
        return results

    def _publish_movies(self, movies_csv_path: str) -> Dict[str, Any]:
        """Publish individual movie events."""
        logger.info(f"Starting movie events publishing from {movies_csv_path}")

        publisher = MovieEventPublisher(self.config)
        try:
            stats = publisher.publish_movies_from_csv(movies_csv_path)
            logger.info(f"Movie events publishing completed: {stats}")
            return stats
        finally:
            publisher.close()

    def _publish_analytics(self, analytics_dir: str) -> Dict[str, Any]:
        """Publish analytics aggregation events."""
        logger.info(f"Starting analytics events publishing from {analytics_dir}")

        publisher = AnalyticsEventPublisher(self.config)
        try:
            stats = publisher.publish_analytics_from_directory(analytics_dir)
            logger.info(f"Analytics events publishing completed: {stats}")
            return stats
        finally:
            publisher.close()

    def get_status(self) -> Dict[str, Any]:
        """Get current status of all publishing jobs."""
        return self.batch_processor.get_all_job_status()

    def export_stats(self, filename: str):
        """Export processing statistics to file."""
        self.batch_processor.export_stats(filename)

    def shutdown(self):
        """Shutdown the orchestrator."""
        logger.info("Shutting down movie events orchestrator")
        self.batch_processor.shutdown()

def _parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Publish movie data to Google Cloud PubSub events"
    )

    parser.add_argument(
        "--movies-csv",
        default="output_large_filtered/movies_facts_denorm.csv",
        help="Path to the movie facts CSV file"
    )

    parser.add_argument(
        "--analytics-dir",
        default="output_large_filtered",
        help="Directory containing analytics CSV files"
    )

    parser.add_argument(
        "--project-id",
        help="Google Cloud project ID (can also set GOOGLE_CLOUD_PROJECT env var)"
    )

    parser.add_argument(
        "--skip-movies",
        action="store_true",
        help="Skip publishing individual movie events"
    )

    parser.add_argument(
        "--skip-analytics",
        action="store_true",
        help="Skip publishing analytics events"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Batch size for publishing (default: 1000)"
    )

    parser.add_argument(
        "--max-workers",
        type=int,
        default=5,
        help="Maximum number of concurrent workers (default: 5)"
    )

    parser.add_argument(
        "--rate-limit",
        type=int,
        default=8000,
        help="Maximum messages per minute (default: 8000)"
    )

    parser.add_argument(
        "--export-stats",
        help="Export statistics to specified JSON file"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs without actually publishing"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    return parser.parse_args()

def main():
    """Main function."""
    args = _parse_args()

    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Create configuration
    config = PubSubConfig()

    # Override configuration with command line arguments
    if args.project_id:
        config.project_id = args.project_id
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.max_workers:
        config.concurrent_workers = args.max_workers
    if args.rate_limit:
        config.messages_per_minute = args.rate_limit

    logger.info(f"Using project ID: {config.project_id}")
    logger.info(f"Configuration: batch_size={config.batch_size}, "
                f"workers={config.concurrent_workers}, "
                f"rate_limit={config.messages_per_minute}")

    # Validate inputs
    publish_movies = not args.skip_movies
    publish_analytics = not args.skip_analytics

    if not publish_movies and not publish_analytics:
        logger.error("Nothing to publish - both movies and analytics are skipped")
        sys.exit(1)

    if args.dry_run:
        logger.info("DRY RUN - Validating inputs without publishing")

        if publish_movies:
            if not os.path.exists(args.movies_csv):
                logger.error(f"Movies CSV file not found: {args.movies_csv}")
                sys.exit(1)
            else:
                logger.info(f"✓ Movies CSV file found: {args.movies_csv}")

        if publish_analytics:
            if not os.path.isdir(args.analytics_dir):
                logger.error(f"Analytics directory not found: {args.analytics_dir}")
                sys.exit(1)
            else:
                logger.info(f"✓ Analytics directory found: {args.analytics_dir}")

        logger.info("✓ Dry run validation passed")
        return

    # Create orchestrator and run
    orchestrator = MovieEventOrchestrator(config)

    try:
        # Publish events
        results = orchestrator.publish_all_events(
            movies_csv_path=args.movies_csv,
            analytics_dir=args.analytics_dir,
            publish_movies=publish_movies,
            publish_analytics=publish_analytics
        )

        # Print summary
        print(f"\n{'='*60}")
        print(f"PUBLISHING SUMMARY")
        print(f"{'='*60}")
        print(f"Total Duration: {results['total_duration']:.2f} seconds")
        print(f"Jobs Completed: {results['jobs_completed']}")

        global_stats = results['global_stats']
        print(f"Total Messages Published: {global_stats['total_messages_published']:,}")
        print(f"Total Messages Failed: {global_stats['total_messages_failed']:,}")
        print(f"Message Success Rate: {global_stats['message_success_rate']:.2%}")
        print(f"Publishing Rate: {global_stats['messages_per_second']:.2f} msg/sec")

        # Print job details
        print(f"\nJOB DETAILS:")
        for job_id, job_details in results['job_details'].items():
            print(f"  {job_id}: {job_details['status']} "
                  f"({job_details['duration']:.2f}s)")
            if job_details.get('stats'):
                stats = job_details['stats']
                if 'total_published' in stats:
                    print(f"    Published: {stats['total_published']:,} messages")
                if 'total_events_published' in stats:
                    print(f"    Published: {stats['total_events_published']:,} events")

        # Export statistics if requested
        if args.export_stats:
            orchestrator.export_stats(args.export_stats)
            print(f"\nStatistics exported to: {args.export_stats}")

        # Check for failures
        if global_stats['total_messages_failed'] > 0:
            logger.warning(f"Some messages failed to publish. "
                          f"Check deadletter topic: {config.deadletter_topic}")

    except Exception as e:
        logger.error(f"Publishing failed: {e}")
        sys.exit(1)

    finally:
        orchestrator.shutdown()

if __name__ == "__main__":
    main()