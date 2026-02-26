# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Suffix Cache Client for updating remote sglang suffix cache servers.

This client communicates with the SuffixCacheServer in sglang to synchronize
suffix tree state across multiple rollout instances in RL training.
"""

import json
import logging
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

logger = logging.getLogger(__name__)


class SuffixCacheClient:
    """
    Client for updating remote suffix cache servers in sglang.

    This client broadcasts cache updates to multiple sglang rollout instances
    to synchronize suffix tree state across the distributed training setup.

    Usage:
        client = SuffixCacheClient(["http://host1:6378", "http://host2:6378"])
        client.update_cache("request_123", [1, 2, 3, 4, 5], prompt_length=2)
    """

    def __init__(
        self,
        server_urls: list[str],
        timeout: float = 5.0,
        max_workers: int = 8,
    ):
        """
        Initialize the cache client.

        Args:
            server_urls: List of server URLs (e.g., ["http://host1:6378", "http://host2:6378"])
            timeout: Request timeout in seconds
            max_workers: Maximum number of concurrent update threads
        """
        self.server_urls = server_urls
        self.timeout = timeout
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

    def update_cache(
        self,
        request_id: str,
        token_ids: list[int],
        prompt_length: int = 0,
    ) -> dict[str, dict]:
        """
        Update the cache on all servers.

        Args:
            request_id: Unique identifier for the request
            token_ids: List of token IDs (prompt + response)
            prompt_length: Length of the prompt portion in token_ids.
                           The remaining tokens are treated as response.
                           Defaults to 0 (all tokens treated as response).

        Returns:
            Dict with results for each server
        """
        results = {}
        data = json.dumps(
            {
                "request_id": request_id,
                "token_ids": token_ids,
                "prompt_length": prompt_length,
            }
        ).encode("utf-8")

        for url in self.server_urls:
            try:
                req = urllib.request.Request(
                    f"{url.rstrip('/')}/update_cache",
                    data=data,
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=self.timeout) as response:
                    results[url] = json.loads(response.read().decode("utf-8"))
            except urllib.error.URLError as e:
                results[url] = {"error": str(e)}
                logger.warning(f"Failed to update cache at {url}: {e}")
            except Exception as e:
                results[url] = {"error": str(e)}
                logger.warning(f"Error updating cache at {url}: {e}")

        return results

    def update_cache_batch(
        self,
        batch_data: list[dict],
    ) -> dict[str, list[dict]]:
        """
        Update cache with a batch of requests on all servers.

        Args:
            batch_data: List of dicts, each containing:
                - request_id: Unique identifier
                - token_ids: List of token IDs
                - prompt_length: Length of prompt portion

        Returns:
            Dict mapping server URL to list of results
        """
        results = {url: [] for url in self.server_urls}

        # Prepare batch request data
        batch_json = json.dumps(batch_data).encode("utf-8")

        for url in self.server_urls:
            try:
                req = urllib.request.Request(
                    f"{url.rstrip('/')}/update_cache_batch",
                    data=batch_json,
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=self.timeout) as response:
                    results[url] = json.loads(response.read().decode("utf-8"))
            except urllib.error.URLError as e:
                results[url] = [{"error": str(e)}] * len(batch_data)
                logger.warning(f"Failed to batch update cache at {url}: {e}")
            except Exception as e:
                results[url] = [{"error": str(e)}] * len(batch_data)
                logger.warning(f"Error batch updating cache at {url}: {e}")

        return results

    def health_check(self) -> dict[str, dict]:
        """
        Check health of all servers.

        Returns:
            Dict with health status for each server
        """
        results = {}
        for url in self.server_urls:
            try:
                req = urllib.request.Request(f"{url.rstrip('/')}/health")
                with urllib.request.urlopen(req, timeout=self.timeout) as response:
                    results[url] = json.loads(response.read().decode("utf-8"))
            except urllib.error.URLError as e:
                results[url] = {"error": str(e)}
            except Exception as e:
                results[url] = {"error": str(e)}

        return results

    def get_stats(self) -> dict[str, dict]:
        """
        Get cache statistics from all servers.

        Returns:
            Dict with stats for each server
        """
        results = {}
        for url in self.server_urls:
            try:
                req = urllib.request.Request(f"{url.rstrip('/')}/stats")
                with urllib.request.urlopen(req, timeout=self.timeout) as response:
                    results[url] = json.loads(response.read().decode("utf-8"))
            except urllib.error.URLError as e:
                results[url] = {"error": str(e)}
            except Exception as e:
                results[url] = {"error": str(e)}

        return results

    def shutdown(self):
        """Shutdown the client and cleanup resources."""
        self._executor.shutdown(wait=True)

    def __del__(self):
        """Ensure cleanup on destruction."""
        self.shutdown()


class AsyncSuffixCacheClient(SuffixCacheClient):
    """
    Async version of SuffixCacheClient using thread pool for non-blocking updates.

    This is useful for RL training where cache updates should not block
    the main training loop.
    """

    def __init__(
        self,
        server_urls: list[str],
        timeout: float = 5.0,
        max_workers: int = 8,
        max_pending_updates: int = 100,
    ):
        """
        Initialize the async cache client.

        Args:
            server_urls: List of server URLs
            timeout: Request timeout in seconds
            max_workers: Maximum number of concurrent update threads
            max_pending_updates: Maximum number of pending async updates
        """
        super().__init__(server_urls, timeout, max_workers)
        self.max_pending_updates = max_pending_updates
        self._pending_futures = []

    def update_cache_async(
        self,
        request_id: str,
        token_ids: list[int],
        prompt_length: int = 0,
    ):
        """
        Asynchronously update the cache on all servers.

        Args:
            request_id: Unique identifier for the request
            token_ids: List of token IDs
            prompt_length: Length of the prompt portion
        """
        # Limit pending futures to prevent memory overflow
        self._wait_for_pending_updates(threshold=self.max_pending_updates)

        future = self._executor.submit(
            self.update_cache,
            request_id=request_id,
            token_ids=token_ids,
            prompt_length=prompt_length,
        )
        self._pending_futures.append(future)

    def update_cache_batch_async(
        self,
        batch_data: list[dict],
    ):
        """
        Asynchronously update cache with a batch of requests.

        Args:
            batch_data: List of dicts containing request_id, token_ids, prompt_length
        """
        self._wait_for_pending_updates(threshold=self.max_pending_updates)

        future = self._executor.submit(
            self.update_cache_batch,
            batch_data=batch_data,
        )
        self._pending_futures.append(future)

    def _wait_for_pending_updates(self, threshold: int):
        """Wait for oldest pending updates if threshold is exceeded."""
        while len(self._pending_futures) >= threshold:
            # Wait for the oldest future to complete
            oldest = self._pending_futures.pop(0)
            try:
                oldest.result(timeout=self.timeout * 2)
            except Exception as e:
                logger.warning(f"Pending update failed: {e}")

    def wait_all(self, timeout: Optional[float] = None):
        """
        Wait for all pending updates to complete.

        Args:
            timeout: Maximum time to wait (None for no limit)
        """
        for future in self._pending_futures:
            try:
                future.result(timeout=timeout)
            except Exception as e:
                logger.warning(f"Pending update failed: {e}")
        self._pending_futures.clear()

    def shutdown(self):
        """Shutdown the client and wait for pending updates."""
        self.wait_all()
        super().shutdown()
