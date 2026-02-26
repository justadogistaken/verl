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
Suffix Cache Manager for distributed suffix cache synchronization in RL training.

This module provides a manager that synchronizes suffix tree state across
multiple rollout instances after each RL training step.
"""

import logging
import socket
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Optional

import numpy as np
import ray

from verl.trainer.ppo.suffix_cache_client import AsyncSuffixCacheClient
from verl.trainer.ppo.utils import Role

if TYPE_CHECKING:
    from omegaconf import DictConfig

logger = logging.getLogger(__name__)


class SuffixCacheManager:
    """
    Manager for distributed suffix cache synchronization in RL training.

    This class handles:
    - Collecting server addresses from rollout workers
    - Updating suffix cache after each training step
    - Async batch updates to avoid blocking training

    Usage:
        cache_manager = SuffixCacheManager(config, actor_rollout_wg, port=6378)
        if cache_manager.enabled:
            # After each rollout step
            cache_manager.update_cache(batch, responses_per_prompt)
        # Cleanup on shutdown
        cache_manager.shutdown()
    """

    def __init__(
        self,
        config: "DictConfig",
        actor_rollout_wg,
        port: int = 6378,
        suffix_cache_config: Optional[dict] = None,
    ):
        """
        Initialize the suffix cache manager.

        Args:
            config: Training configuration
            actor_rollout_wg: Actor rollout worker group
            port: Port number for suffix cache servers
            suffix_cache_config: Additional config for suffix cache
        """
        self.config = config
        self.actor_rollout_wg = actor_rollout_wg
        self.port = port
        self.suffix_cache_config = suffix_cache_config or {}

        # Internal state
        self._cache_client: Optional[AsyncSuffixCacheClient] = None
        self._server_urls: list[str] = []
        self._executor: Optional[ThreadPoolExecutor] = None
        self._max_workers = self.suffix_cache_config.get("max_workers", 4)
        self._max_pending_updates = self.suffix_cache_config.get("max_pending_updates", 100)
        self._update_timeout = self.suffix_cache_config.get("update_timeout", 5.0)

        # Check if suffix cache is enabled
        self._enabled = self._should_enable()

        if self._enabled:
            self._initialize()

    def _should_enable(self) -> bool:
        """Check if suffix cache should be enabled based on configuration."""
        # Check if rollout has suffix cache enabled
        rollout_config = self.config.actor_rollout_ref.rollout

        # Check for explicit enable flag
        suffix_cache_enabled = rollout_config.get("suffix_cache", {}).get("enable", False)

        # Also check if using sglang rollout (which supports suffix cache)
        rollout_name = rollout_config.get("name", "")
        is_sglang = rollout_name == "sglang"

        return suffix_cache_enabled and is_sglang

    def _initialize(self):
        """Initialize cache client with server addresses from rollout workers."""
        # Get server addresses from rollout workers
        self._server_urls = self._get_server_addresses()

        if not self._server_urls:
            logger.warning("No suffix cache server addresses found, disabling suffix cache manager")
            self._enabled = False
            return

        # Initialize async cache client
        self._cache_client = AsyncSuffixCacheClient(
            server_urls=self._server_urls,
            timeout=self._update_timeout,
            max_workers=self._max_workers,
            max_pending_updates=self._max_pending_updates,
        )

        # Thread pool for async updates from trainer
        self._executor = ThreadPoolExecutor(max_workers=self._max_workers)

        logger.info(f"SuffixCacheManager initialized with {len(self._server_urls)} servers on port {self.port}")
        logger.info(f"Server URLs: {self._server_urls}")

    def _get_server_addresses(self) -> list[str]:
        """
        Get server addresses from rollout workers.

        Returns:
            List of server URLs in format 'http://host:port'
        """
        server_urls = []

        try:
            # Get worker info from the worker group
            # The actor_rollout_wg contains the rollout workers
            # Each worker runs an sglang server that can receive cache updates

            # Try to get addresses from worker group
            if hasattr(self.actor_rollout_wg, "get_rollout_server_addresses"):
                # Custom method to get server addresses
                addresses = ray.get(self.actor_rollout_wg.get_rollout_server_addresses.remote())
                for addr in addresses:
                    server_urls.append(f"http://{addr}:{self.port}")
            else:
                # Fallback: construct addresses from Ray node info
                node_ips = self._get_node_ips_from_workers()
                for ip in node_ips:
                    # Handle IPv6 addresses
                    if ":" in ip and not ip.startswith("["):
                        server_urls.append(f"http://[{ip}]:{self.port}")
                    else:
                        server_urls.append(f"http://{ip}:{self.port}")

        except Exception as e:
            logger.error(f"Failed to get server addresses: {e}")

        return server_urls

    def _get_node_ips_from_workers(self) -> list[str]:
        """
        Get unique node IPs from worker placement.

        Returns:
            List of unique node IP addresses
        """
        node_ips = set()

        try:
            # Get placement group info from worker group
            if hasattr(self.actor_rollout_wg, "workers"):
                for worker in self.actor_rollout_wg.workers:
                    # Get the node IP where this worker is running
                    node_id = ray.get(worker.__ray_actor__.get_node_id.remote())
                    node_info = ray._private.state.state.node_table()
                    for node in node_info:
                        if node.get("NodeID") == node_id:
                            node_ip = node.get("NodeManagerAddress", "")
                            if node_ip:
                                node_ips.add(node_ip)
                            break

            # Fallback: use Ray's internal state to get all node IPs
            if not node_ips:
                for node in ray._private.state.state.node_table():
                    node_ip = node.get("NodeManagerAddress", "")
                    if node_ip:
                        node_ips.add(node_ip)

        except Exception as e:
            logger.warning(f"Failed to get node IPs from workers: {e}")
            # Ultimate fallback: use localhost
            node_ips.add("127.0.0.1")

        return list(node_ips)

    def update_cache(
        self,
        batch,
        responses_per_prompt: int = 1,
    ):
        """
        Update the suffix cache with new generation results asynchronously.

        This method extracts prompts and responses from the batch and submits them
        to the cache client for async processing. The cache is updated across all
        cache servers in a distributed manner.

        Args:
            batch: DataProto containing prompts, responses, and attention masks
            responses_per_prompt: Number of responses generated per prompt
        """
        if not self._enabled or self._cache_client is None:
            return

        try:
            # Extract token sequences from batch
            batch_data = self._extract_batch_data(batch, responses_per_prompt)

            if not batch_data:
                return

            # Submit async update
            self._cache_client.update_cache_batch_async(batch_data)

        except Exception as e:
            logger.warning(f"Failed to update suffix cache: {e}")

    def _extract_batch_data(
        self,
        batch,
        responses_per_prompt: int = 1,
    ) -> list[dict]:
        """
        Extract prompt and response data from batch for cache update.

        Args:
            batch: DataProto with prompts, responses, attention_mask
            responses_per_prompt: Number of responses per prompt

        Returns:
            List of dicts with request_id, token_ids, prompt_length
        """
        batch_data = []

        # Get tensors from batch
        prompts = batch.batch["prompts"]  # (batch_size, prompt_length)
        responses = batch.batch["responses"]  # (batch_size, response_length)
        attention_mask = batch.batch["attention_mask"]  # (batch_size, total_length)

        response_length = responses.shape[-1]

        # Split attention mask into prompt and response parts
        prompt_mask = attention_mask[:, :-response_length]
        response_mask = attention_mask[:, -response_length:]

        # Calculate actual lengths (excluding padding)
        prompt_lengths = prompt_mask.sum(-1).cpu().tolist()
        response_lengths = response_mask.sum(-1).cpu().tolist()

        # Convert tensors to lists
        prompts_list = prompts.cpu().tolist()
        responses_list = responses.cpu().tolist()

        batch_size = len(prompts_list)

        for i in range(batch_size):
            prompt_len = int(prompt_lengths[i])
            response_len = int(response_lengths[i])

            if prompt_len == 0 or response_len == 0:
                continue

            # Extract actual tokens (without padding)
            prompt_tokens = prompts_list[i][:prompt_len]
            response_tokens = responses_list[i][:response_len]

            # Combine prompt and response
            token_ids = prompt_tokens + response_tokens

            # Generate unique request ID
            request_id = str(uuid.uuid4())

            batch_data.append({
                "request_id": request_id,
                "token_ids": token_ids,
                "prompt_length": prompt_len,
            })

        return batch_data

    def wait_for_updates(self, timeout: Optional[float] = None):
        """
        Wait for all pending cache updates to complete.

        Args:
            timeout: Maximum time to wait (None for no limit)
        """
        if self._cache_client is not None:
            self._cache_client.wait_all(timeout=timeout)

    def health_check(self) -> dict:
        """
        Check health of all cache servers.

        Returns:
            Dict with health status for each server
        """
        if self._cache_client is None:
            return {"error": "Cache client not initialized"}
        return self._cache_client.health_check()

    def get_stats(self) -> dict:
        """
        Get cache statistics from all servers.

        Returns:
            Dict with stats for each server
        """
        if self._cache_client is None:
            return {"error": "Cache client not initialized"}
        return self._cache_client.get_stats()

    @property
    def enabled(self) -> bool:
        """Check if suffix cache manager is enabled."""
        return self._enabled

    @property
    def server_urls(self) -> list[str]:
        """Get the list of server URLs."""
        return self._server_urls.copy()

    def shutdown(self):
        """Clean up cache client and executor resources."""
        if self._cache_client is not None:
            self._cache_client.shutdown()
            self._cache_client = None

        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

        logger.info("SuffixCacheManager shutdown complete")

    def __del__(self):
        """Ensure cleanup on destruction."""
        self.shutdown()
