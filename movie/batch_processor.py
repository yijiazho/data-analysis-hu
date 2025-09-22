"""
Batch processor with advanced error handling and monitoring for PubSub events.
"""
import json
import logging
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional, Callable
from enum import Enum
import queue

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class JobStatus(Enum):
    """Status of a processing job."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class JobResult:
    """Result of a processing job."""
    job_id: str
    status: JobStatus
    start_time: float
    end_time: Optional[float] = None
    stats: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    @property
    def duration(self) -> float:
        """Calculate job duration."""
        if self.end_time:
            return self.end_time - self.start_time
        return time.time() - self.start_time

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        result['status'] = self.status.value
        result['duration'] = self.duration
        return result

class RateLimiter:
    """Rate limiter for controlling message throughput."""

    def __init__(self, max_messages_per_minute: int):
        """Initialize rate limiter."""
        self.max_messages_per_minute = max_messages_per_minute
        self.messages_sent = []
        self.lock = threading.Lock()

    def wait_if_needed(self):
        """Wait if rate limit would be exceeded."""
        with self.lock:
            current_time = time.time()

            # Remove messages older than 1 minute
            cutoff_time = current_time - 60
            self.messages_sent = [t for t in self.messages_sent if t > cutoff_time]

            # Check if we need to wait
            if len(self.messages_sent) >= self.max_messages_per_minute:
                # Calculate wait time until oldest message is older than 1 minute
                wait_time = 60 - (current_time - self.messages_sent[0])
                if wait_time > 0:
                    logger.info(f"Rate limiting: waiting {wait_time:.2f} seconds")
                    time.sleep(wait_time)
                    # Re-clean the list after waiting
                    current_time = time.time()
                    cutoff_time = current_time - 60
                    self.messages_sent = [t for t in self.messages_sent if t > cutoff_time]

            # Record this message
            self.messages_sent.append(current_time)

class BatchProcessor:
    """Batch processor with error handling, monitoring, and rate limiting."""

    def __init__(self, max_workers: int = 5, max_messages_per_minute: int = 8000):
        """Initialize batch processor."""
        self.max_workers = max_workers
        self.rate_limiter = RateLimiter(max_messages_per_minute)
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.job_queue = queue.Queue()
        self.active_jobs: Dict[str, JobResult] = {}
        self.completed_jobs: List[JobResult] = []
        self.shutdown_event = threading.Event()

        # Statistics
        self.global_stats = {
            'total_jobs_submitted': 0,
            'total_jobs_completed': 0,
            'total_jobs_failed': 0,
            'total_messages_processed': 0,
            'total_messages_published': 0,
            'total_messages_failed': 0,
            'start_time': time.time()
        }

    def submit_job(
        self,
        job_id: str,
        job_function: Callable[[], Dict[str, Any]],
        *args,
        **kwargs
    ) -> str:
        """
        Submit a job for processing.

        Args:
            job_id: Unique identifier for the job
            job_function: Function to execute
            *args, **kwargs: Arguments to pass to the function

        Returns:
            Job ID
        """
        if job_id in self.active_jobs:
            raise ValueError(f"Job {job_id} is already active")

        job_result = JobResult(
            job_id=job_id,
            status=JobStatus.PENDING,
            start_time=time.time()
        )

        self.active_jobs[job_id] = job_result
        self.global_stats['total_jobs_submitted'] += 1

        # Submit to executor
        future = self.executor.submit(self._execute_job, job_result, job_function, *args, **kwargs)

        logger.info(f"Submitted job {job_id}")
        return job_id

    def _execute_job(
        self,
        job_result: JobResult,
        job_function: Callable[[], Dict[str, Any]],
        *args,
        **kwargs
    ):
        """Execute a job with error handling."""
        job_result.status = JobStatus.RUNNING
        logger.info(f"Starting job {job_result.job_id}")

        try:
            # Apply rate limiting before execution
            self.rate_limiter.wait_if_needed()

            # Execute the job function
            stats = job_function(*args, **kwargs)

            # Update job result
            job_result.status = JobStatus.COMPLETED
            job_result.stats = stats
            job_result.end_time = time.time()

            # Update global statistics
            self._update_global_stats(stats)
            self.global_stats['total_jobs_completed'] += 1

            logger.info(f"Job {job_result.job_id} completed successfully in {job_result.duration:.2f}s")

        except Exception as e:
            job_result.status = JobStatus.FAILED
            job_result.error = str(e)
            job_result.end_time = time.time()
            self.global_stats['total_jobs_failed'] += 1

            logger.error(f"Job {job_result.job_id} failed after {job_result.duration:.2f}s: {e}")

        finally:
            # Move from active to completed
            if job_result.job_id in self.active_jobs:
                del self.active_jobs[job_result.job_id]
            self.completed_jobs.append(job_result)

    def _update_global_stats(self, job_stats: Dict[str, Any]):
        """Update global statistics with job results."""
        if 'total_processed' in job_stats:
            self.global_stats['total_messages_processed'] += job_stats['total_processed']
        if 'total_published' in job_stats:
            self.global_stats['total_messages_published'] += job_stats['total_published']
        if 'total_failed' in job_stats:
            self.global_stats['total_messages_failed'] += job_stats['total_failed']

        # Handle analytics-specific stats
        if 'total_events_published' in job_stats:
            self.global_stats['total_messages_published'] += job_stats['total_events_published']

    def wait_for_jobs(self, timeout: Optional[float] = None) -> List[JobResult]:
        """
        Wait for all active jobs to complete.

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            List of completed job results
        """
        start_time = time.time()

        while self.active_jobs:
            if timeout and (time.time() - start_time) > timeout:
                logger.warning(f"Timeout waiting for jobs. {len(self.active_jobs)} jobs still active")
                break

            time.sleep(0.5)

        return self.completed_jobs.copy()

    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific job."""
        # Check active jobs
        if job_id in self.active_jobs:
            return self.active_jobs[job_id].to_dict()

        # Check completed jobs
        for job in self.completed_jobs:
            if job.job_id == job_id:
                return job.to_dict()

        return None

    def get_all_job_status(self) -> Dict[str, Any]:
        """Get status of all jobs."""
        active = {job_id: job.to_dict() for job_id, job in self.active_jobs.items()}
        completed = [job.to_dict() for job in self.completed_jobs]

        return {
            'active_jobs': active,
            'completed_jobs': completed,
            'global_stats': self.get_global_stats()
        }

    def get_global_stats(self) -> Dict[str, Any]:
        """Get global processing statistics."""
        stats = self.global_stats.copy()
        stats['total_duration'] = time.time() - stats['start_time']

        # Calculate rates
        if stats['total_duration'] > 0:
            stats['jobs_per_second'] = stats['total_jobs_completed'] / stats['total_duration']
            stats['messages_per_second'] = stats['total_messages_published'] / stats['total_duration']
        else:
            stats['jobs_per_second'] = 0
            stats['messages_per_second'] = 0

        # Calculate success rates
        total_jobs = stats['total_jobs_completed'] + stats['total_jobs_failed']
        if total_jobs > 0:
            stats['job_success_rate'] = stats['total_jobs_completed'] / total_jobs
        else:
            stats['job_success_rate'] = 0

        total_messages = stats['total_messages_published'] + stats['total_messages_failed']
        if total_messages > 0:
            stats['message_success_rate'] = stats['total_messages_published'] / total_messages
        else:
            stats['message_success_rate'] = 1.0

        return stats

    def cancel_job(self, job_id: str) -> bool:
        """Cancel an active job."""
        if job_id in self.active_jobs:
            # Note: This is a simple implementation. In practice, you'd need
            # more sophisticated cancellation handling for running jobs.
            job = self.active_jobs[job_id]
            if job.status == JobStatus.PENDING:
                job.status = JobStatus.CANCELLED
                job.end_time = time.time()
                del self.active_jobs[job_id]
                self.completed_jobs.append(job)
                logger.info(f"Cancelled job {job_id}")
                return True

        return False

    def shutdown(self, wait: bool = True):
        """Shutdown the batch processor."""
        logger.info("Shutting down batch processor...")
        self.shutdown_event.set()

        if wait:
            # Wait for active jobs to complete
            self.wait_for_jobs(timeout=60)

        # Shutdown executor
        self.executor.shutdown(wait=wait)
        logger.info("Batch processor shutdown complete")

    def export_stats(self, filename: str):
        """Export processing statistics to JSON file."""
        stats = {
            'global_stats': self.get_global_stats(),
            'job_results': [job.to_dict() for job in self.completed_jobs],
            'export_time': time.time()
        }

        with open(filename, 'w') as f:
            json.dump(stats, f, indent=2)

        logger.info(f"Statistics exported to {filename}")

def main():
    """Example usage of the batch processor."""
    processor = BatchProcessor(max_workers=3, max_messages_per_minute=100)

    # Example job function
    def example_job(job_name: str, duration: float):
        logger.info(f"Running {job_name} for {duration} seconds")
        time.sleep(duration)
        return {
            'total_processed': 100,
            'total_published': 95,
            'total_failed': 5,
            'job_name': job_name
        }

    try:
        # Submit some test jobs
        for i in range(5):
            job_id = f"test_job_{i}"
            processor.submit_job(job_id, example_job, f"Job {i}", i + 1)

        # Wait for completion
        results = processor.wait_for_jobs(timeout=30)

        # Print results
        print(f"Completed {len(results)} jobs")
        print(f"Global stats: {processor.get_global_stats()}")

        # Export statistics
        processor.export_stats("batch_processing_stats.json")

    finally:
        processor.shutdown()

if __name__ == "__main__":
    main()