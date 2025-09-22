#!/usr/bin/env python3
"""
Setup script for creating PubSub topics and subscriptions for movie events.
"""
import argparse
import logging
from google.cloud import pubsub_v1
from google.api_core import exceptions

from pubsub_config import PubSubConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PubSubSetup:
    """Setup PubSub topics and subscriptions for movie events."""

    def __init__(self, config: PubSubConfig):
        """Initialize the setup with configuration."""
        self.config = config
        self.publisher = pubsub_v1.PublisherClient()
        self.subscriber = pubsub_v1.SubscriberClient()

    def create_topics(self) -> bool:
        """Create all required PubSub topics."""
        logger.info("Creating PubSub topics...")

        topics_config = [
            {
                'name': self.config.raw_events_topic,
                'path': self.config.raw_events_topic_path,
                'description': 'Individual movie events'
            },
            {
                'name': self.config.analytics_topic,
                'path': self.config.analytics_topic_path,
                'description': 'Analytics aggregation events'
            },
            {
                'name': self.config.deadletter_topic,
                'path': self.config.deadletter_topic_path,
                'description': 'Failed/invalid events'
            }
        ]

        success = True

        for topic_config in topics_config:
            try:
                # Try to create the topic
                topic = self.publisher.create_topic(request={"name": topic_config['path']})
                logger.info(f"✓ Created topic: {topic_config['name']}")

            except exceptions.AlreadyExists:
                logger.info(f"✓ Topic already exists: {topic_config['name']}")

            except Exception as e:
                logger.error(f"✗ Failed to create topic {topic_config['name']}: {e}")
                success = False

        return success

    def create_example_subscriptions(self) -> bool:
        """Create example subscriptions for testing and common use cases."""
        logger.info("Creating example subscriptions...")

        subscriptions_config = [
            {
                'name': 'movie-events-all',
                'topic': self.config.raw_events_topic_path,
                'filter': None,
                'description': 'All movie events'
            },
            {
                'name': 'movie-events-high-rated',
                'topic': self.config.raw_events_topic_path,
                'filter': 'attributes.ratingTier="high"',
                'description': 'High-rated movies only'
            },
            {
                'name': 'movie-events-blockbusters',
                'topic': self.config.raw_events_topic_path,
                'filter': 'attributes.revenueTier="blockbuster"',
                'description': 'Blockbuster movies only'
            },
            {
                'name': 'movie-events-recent',
                'topic': self.config.raw_events_topic_path,
                'filter': 'attributes.decade="2020s" OR attributes.decade="2010s"',
                'description': 'Recent movies (2010s and 2020s)'
            },
            {
                'name': 'analytics-events-all',
                'topic': self.config.analytics_topic_path,
                'filter': None,
                'description': 'All analytics events'
            },
            {
                'name': 'analytics-events-genre',
                'topic': self.config.analytics_topic_path,
                'filter': 'attributes.aggregationType="by_genre"',
                'description': 'Genre analytics only'
            },
            {
                'name': 'deadletter-monitor',
                'topic': self.config.deadletter_topic_path,
                'filter': None,
                'description': 'Monitor failed events'
            }
        ]

        success = True

        for sub_config in subscriptions_config:
            try:
                subscription_path = self.subscriber.subscription_path(
                    self.config.project_id, sub_config['name']
                )

                request = {
                    "name": subscription_path,
                    "topic": sub_config['topic'],
                }

                # Add filter if specified
                if sub_config['filter']:
                    request["filter"] = sub_config['filter']

                # Set message retention
                request["message_retention_duration"] = {"seconds": 604800}  # 7 days

                # Create subscription
                subscription = self.subscriber.create_subscription(request=request)
                logger.info(f"✓ Created subscription: {sub_config['name']}")

            except exceptions.AlreadyExists:
                logger.info(f"✓ Subscription already exists: {sub_config['name']}")

            except Exception as e:
                logger.error(f"✗ Failed to create subscription {sub_config['name']}: {e}")
                success = False

        return success

    def list_topics(self):
        """List all topics in the project."""
        logger.info("Listing topics...")
        project_path = f"projects/{self.config.project_id}"

        try:
            topics = list(self.publisher.list_topics(request={"project": project_path}))
            if topics:
                for topic in topics:
                    logger.info(f"  - {topic.name}")
            else:
                logger.info("  No topics found")

        except Exception as e:
            logger.error(f"Failed to list topics: {e}")

    def list_subscriptions(self):
        """List all subscriptions in the project."""
        logger.info("Listing subscriptions...")
        project_path = f"projects/{self.config.project_id}"

        try:
            subscriptions = list(self.subscriber.list_subscriptions(request={"project": project_path}))
            if subscriptions:
                for subscription in subscriptions:
                    logger.info(f"  - {subscription.name}")
                    logger.info(f"    Topic: {subscription.topic}")
                    if hasattr(subscription, 'filter') and subscription.filter:
                        logger.info(f"    Filter: {subscription.filter}")
            else:
                logger.info("  No subscriptions found")

        except Exception as e:
            logger.error(f"Failed to list subscriptions: {e}")

    def delete_topics(self, confirm: bool = False) -> bool:
        """Delete all movie-related topics (dangerous!)."""
        if not confirm:
            logger.warning("Delete operation requires confirmation. Use --confirm flag.")
            return False

        logger.warning("DELETING all movie-related topics...")

        topics_to_delete = [
            self.config.raw_events_topic_path,
            self.config.analytics_topic_path,
            self.config.deadletter_topic_path
        ]

        success = True

        for topic_path in topics_to_delete:
            try:
                self.publisher.delete_topic(request={"topic": topic_path})
                logger.info(f"✓ Deleted topic: {topic_path}")

            except exceptions.NotFound:
                logger.info(f"✓ Topic not found (already deleted): {topic_path}")

            except Exception as e:
                logger.error(f"✗ Failed to delete topic {topic_path}: {e}")
                success = False

        return success

def _parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Setup PubSub topics and subscriptions for movie events"
    )

    parser.add_argument(
        "--project-id",
        help="Google Cloud project ID (can also set GOOGLE_CLOUD_PROJECT env var)"
    )

    parser.add_argument(
        "--create-topics",
        action="store_true",
        help="Create PubSub topics"
    )

    parser.add_argument(
        "--create-subscriptions",
        action="store_true",
        help="Create example subscriptions"
    )

    parser.add_argument(
        "--list-topics",
        action="store_true",
        help="List all topics"
    )

    parser.add_argument(
        "--list-subscriptions",
        action="store_true",
        help="List all subscriptions"
    )

    parser.add_argument(
        "--delete-topics",
        action="store_true",
        help="Delete all movie-related topics (dangerous!)"
    )

    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Confirm dangerous operations"
    )

    parser.add_argument(
        "--setup-all",
        action="store_true",
        help="Create topics and subscriptions (equivalent to --create-topics --create-subscriptions)"
    )

    return parser.parse_args()

def main():
    """Main function."""
    args = _parse_args()

    # Create configuration
    config = PubSubConfig()
    if args.project_id:
        config.project_id = args.project_id

    logger.info(f"Using project ID: {config.project_id}")

    # Create setup instance
    setup = PubSubSetup(config)

    # Handle setup-all option
    if args.setup_all:
        args.create_topics = True
        args.create_subscriptions = True

    # Execute requested operations
    if args.create_topics:
        success = setup.create_topics()
        if not success:
            logger.error("Topic creation failed")
            return 1

    if args.create_subscriptions:
        success = setup.create_example_subscriptions()
        if not success:
            logger.error("Subscription creation failed")
            return 1

    if args.list_topics:
        setup.list_topics()

    if args.list_subscriptions:
        setup.list_subscriptions()

    if args.delete_topics:
        success = setup.delete_topics(confirm=args.confirm)
        if not success:
            logger.error("Topic deletion failed")
            return 1

    # If no action specified, show help
    if not any([args.create_topics, args.create_subscriptions, args.list_topics,
                args.list_subscriptions, args.delete_topics, args.setup_all]):
        logger.info("No action specified. Use --help for options.")
        logger.info("Quick start: python setup_pubsub_topics.py --setup-all")

    return 0

if __name__ == "__main__":
    exit(main())