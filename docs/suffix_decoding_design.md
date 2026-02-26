# Suffix Decoding For Accelerating RL Rollout

## 动态开关 Suffix Decoding 设计文档

---

## 1. 背景介绍

### 1.1 Suffix Decoding 简介

Suffix Decoding 是一种基于后缀树（Suffix Tree）的推测解码算法，通过缓存历史请求的 token 序列模式，在 decode 阶段预测可能的后继 token，从而实现一次前向传播生成多个 token 的效果。

**核心优势：**
- 接受率高（基于真实数据模式预测）
- 内存开销小（仅存储 token 序列索引）
- 适合重复模式多的场景（如代码补全、结构化输出）

### 1.2 问题背景

在实际生产环境中，请求负载是动态变化的：

- **低负载时（batch size 小）：** 推测解码收益高，接受率可达 30-50%。
- **高负载时（batch size 大）：** 推测解码收益低，反而因为 draft token 的额外计算导致吞吐量下降。

---

**实测数据（Qwen2.5-7B，draft_token_num=24）：**

| Batch Size | Spec 状态始终禁用 吞吐量 (token/s) | Spec 状态始终启用 吞吐量 (token/s) |
|------------|-----------------------------------|-----------------------------------|
| 4          | ~780                              | ~2000                             |
| 8          | ~1500                             | ~2500                             |
| 32         | ~4500                             | ~2500                             |
| 64         | ~6000                             | ~1700                             |

**结论：** 当 batch size 超过某个阈值时，禁用推测解码反而能获得更好的性能。

---

## 2. 核心设计

### 2.1 设计目标

1. **自适应启停：** 根据 batch size 动态启用/禁用推测解码。
2. **无状态切换：** 确保 spec ↔ non-spec 状态切换时不会导致错误或内存泄漏。
3. **CUDA Graph 兼容：** 支持两种模式的 CUDA Graph 分别捕获和回放。
4. **外部接口：** 提供 API 让外部框架主动更新 suffix tree。

### 2.2 架构概览

```
┌─────────────────────────────────────────────────────────────────┐
│                         Scheduler                                │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ update_running_batch()                                      ││
│  │   └── prepare_for_decode()                                  ││
│  │         └── 检查 spec_algorithm 决定是否跳过                ││
│  └─────────────────────────────────────────────────────────────┘│
│                              ↓                                   │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ run_batch()                                                 ││
│  │   └── model_worker.forward_batch_generation()               ││
│  │         └── NgramWorker / SuffixWorker                      ││
│  │               ├── 检查 batch_size vs threshold              ││
│  │               ├── 动态设置 spec_algorithm                   ││
│  │               └── 调用 target_worker 或 spec 逻辑           ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    CUDA Graph Runner                             │
│  ┌─────────────────────┐    ┌─────────────────────┐             │
│  │   Spec Graphs       │    │  Non-Spec Graphs    │             │
│  │  (TARGET_VERIFY)    │    │     (DECODE)        │             │
│  │  positions = bs*24  │    │  positions = bs     │             │
│  └─────────────────────┘    └─────────────────────┘             │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. 实现细节

### 3.1 配置参数

```python
# server_args.py
class ServerArgs:
    # 禁用推测解码的 batch size 阈值
    speculative_disable_batch_size_threshold: int = 0  # 0 表示禁用自适应功能

    # Suffix cache server 端口（用于外部更新）
    speculative_suffix_cache_server_port: int = 0  # 0 表示不启动 server
```

### 3.2 动态选择 CUDA Graph

**关键点：** CUDA Graph 在捕获时就固定了 `num_tokens_per_bs`：
- **Spec 模式：** `num_tokens_per_bs = draft_token_num`（如 24）
- **Non-Spec 模式：** `num_tokens_per_bs = 1`

**实现（cuda_graph_runner.py）：**

```python
class CudaGraphRunner:
    def __init__(self, ...):
        # 检测是否启用自适应推测解码
        self.enable_adaptive_spec = (
            (model_runner.spec_algorithm.is_ngram() or
             model_runner.spec_algorithm.is_suffix()) and
            model_runner.server_args.speculative_disable_batch_size_threshold > 0
        )

        if self.enable_adaptive_spec:
            # 额外捕获 non-spec graphs
            self.graphs_no_spec = {}
            self.output_buffers_no_spec = {}
            self.num_tokens_per_bs_no_spec = 1

    def can_run(self, forward_batch):
        # 根据当前 spec_algorithm 选择 graph 集合
        use_no_spec_graph = (
            self.enable_adaptive_spec and
            forward_batch.spec_algorithm == SpeculativeAlgorithm.NONE
        )
        # ...

    def replay(self, forward_batch, ...):
        # 确定使用哪组 graph
        self.use_no_spec_graph = (
            self.enable_adaptive_spec and
            forward_batch.spec_algorithm == SpeculativeAlgorithm.NONE
        )

        # 计算正确的 token 数量
        if self.use_no_spec_graph:
            raw_num_token = raw_bs * self.num_tokens_per_bs_no_spec  # = bs
        else:
            raw_num_token = raw_bs * self.num_tokens_per_bs  # = bs * 24
```

### 3.3 forward_batch_generation 处理

**核心挑战：** 执行顺序导致的状态不一致。

**执行顺序：**
1. `scheduler.update_running_batch()`
2. └── `batch.prepare_for_decode()`（根据 `spec_algorithm` 决定是否跳过）
3. `scheduler.run_batch()`
4. └── `worker.forward_batch_generation()`（此时才能知道实际 batch_size）

**关键实现（ngram_worker.py）：**

```python
def forward_batch_generation(self, batch: ScheduleBatch) -> GenerationBatchResult:
    bs = batch.batch_size()

    # 1. Prefill 模式直接转发
    if batch.forward_mode.is_extend():
        return self.target_worker.forward_batch_generation(...)

    # 2. 判断是否需要禁用 spec
    should_disable_spec = (
        self.disable_batch_size_threshold > 0 and
        bs > self.disable_batch_size_threshold
    )

    # 3. 判断是否从 no-spec 切换到 spec
    was_spec_disabled = batch.spec_algorithm.is_none()
    transitioning_to_spec = was_spec_disabled and not should_disable_spec

    # 4. No-Spec 分支（batch_size > 阈值）
    if should_disable_spec:
        # 关键：设置正确的状态
        batch.forward_mode = ForwardMode.DECODE
        batch.spec_algorithm = SpeculativeAlgorithm.NONE
        batch.spec_info = None  # 清除旧的 spec 信息

        # 手动执行 decode 准备（因为 prepare_for_decode 跳过了）
        if batch.input_ids is None or len(batch.input_ids) != bs:
            # 处理 merged batch（新请求 + 旧请求）
            batch.input_ids = torch.tensor([
                req.output_ids[-1] if req.output_ids else req.origin_input_ids[-1]
                for req in batch.reqs
            ], device=batch.device)
            batch.out_cache_loc = alloc_for_decode(batch, token_per_req=1)
            # 更新 seq_lens 等...

        return self.target_worker.forward_batch_generation(...)

    # 5. Spec 分支（batch_size <= 阈值）
    if transitioning_to_spec:
        # 从 no-spec 切换到 spec，需要 undo 之前的状态
        batch.seq_lens.sub_(1)
        batch.seq_lens_cpu.sub_(1)
        for req in batch.reqs:
            req.kv_committed_len -= 1
            req.kv_allocated_len -= 1

        # 释放之前分配的 KV cache
        if batch.out_cache_loc is not None:
            batch.token_to_kv_pool_allocator.free(batch.out_cache_loc)

        batch.input_ids = None
        batch.out_cache_loc = None

    # 6. 正常 spec 流程
    self._prepare_for_speculative_decoding(batch)
    # ...
```

---

## 4. 外部接口设计

### 4.1 Suffix Cache Server

**用途：** 允许外部框架（如 RL Rollout）主动更新 suffix tree。

```python
class SuffixCacheServer:
    def __init__(self, cache_adapter, port):
        self.cache = cache_adapter
        self.app = FastAPI()

        @self.app.post("/update_cache")
        async def update_cache(request: CacheUpdateRequest):
            """
            更新 suffix cache

            Args:
                request_id: 请求唯一标识
                token_ids: 完整的 token 序列
                prompt_length: prompt 长度（用于区分 prompt 和 response）
            """
            prompt = request.token_ids[:request.prompt_length]
            response = request.token_ids[request.prompt_length:]
            self.cache._add_completed_request_to_cache(
                request.request_id, prompt, response
            )
            return {"status": "ok"}

        @self.app.post("/update_cache_batch")
        async def update_cache_batch(requests: list[CacheUpdateRequest]):
            """批量更新 suffix cache"""
            for req in requests:
                await update_cache(req)
            return {"status": "ok", "count": len(requests)}

        @self.app.get("/health")
        async def health():
            return {"status": "healthy"}

        @self.app.get("/stats")
        async def stats():
            return {
                "active_requests": len(self.cache.req_state),
                "cached_requests": len(self.cache.suffix_cache.cached_requests),
            }
```

---

## 5. verl 集成

### 5.1 问题背景

在 RL 训练场景中，每个 rollout worker 独立维护自己的 suffix tree。当请求被分发到某个 worker 时，该请求的轨迹只会更新到执行该请求的 worker 本地 suffix tree。这导致：

1. **缓存命中率低：** 不同 worker 之间无法共享生成的序列模式
2. **冗余计算：** 相似 prompt 的 response 无法被复用
3. **接受率受限：** 每个 worker 的历史数据有限

### 5.2 设计目标

实现跨 rollout worker 的 suffix tree 同步机制：
- 每个 RL step 完成后，将生成的完整 prompt-response 序列广播到所有 worker
- 利用 Suffix Cache Server 的 HTTP 接口进行异步更新
- 不阻塞主训练流程

### 5.3 架构设计

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          verl RayPPOTrainer                              │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                        fit() training loop                        │  │
│  │  ┌─────────────────────────────────────────────────────────────┐  │  │
│  │  │ 1. generate_sequences()                                     │  │  │
│  │  │ 2. compute_reward()                                         │  │  │
│  │  │ 3. compute_advantage()                                      │  │  │
│  │  │ 4. update_actor() / update_critic()                         │  │  │
│  │  │ 5. suffix_cache_manager.update_cache(batch)  ← 新增         │  │  │
│  │  │    └── async broadcast to all workers                       │  │  │
│  │  └─────────────────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
                              ↓ async broadcast
┌──────────────────────────────────────────────────────────────────────────┐
│                      Suffix Cache Servers                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │  Worker 0    │  │  Worker 1    │  │  Worker 2    │  │  Worker N    │ │
│  │  :6378       │  │  :6378       │  │  :6378       │  │  :6378       │ │
│  │  /update_    │  │  /update_    │  │  /update_    │  │  /update_    │ │
│  │   cache      │  │   cache      │  │   cache      │  │   cache      │ │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘ │
└──────────────────────────────────────────────────────────────────────────┘
```

### 5.4 核心组件

#### 5.4.1 SuffixCacheClient

HTTP 客户端，负责向多个 sglang suffix cache server 发送更新请求：

```python
# verl/trainer/ppo/suffix_cache_client.py

class SuffixCacheClient:
    """Client for updating remote suffix cache servers in sglang."""

    def __init__(self, server_urls: list[str], timeout: float = 5.0):
        self.server_urls = server_urls
        self.timeout = timeout

    def update_cache(self, request_id: str, token_ids: list[int], prompt_length: int = 0) -> dict:
        """Update cache on all servers."""
        data = json.dumps({
            "request_id": request_id,
            "token_ids": token_ids,
            "prompt_length": prompt_length,
        }).encode("utf-8")

        results = {}
        for url in self.server_urls:
            req = urllib.request.Request(
                f"{url}/update_cache",
                data=data,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                results[url] = json.loads(response.read())
        return results

    def update_cache_batch(self, batch_data: list[dict]) -> dict:
        """Batch update cache on all servers."""
        # ... batch update implementation
```

#### 5.4.2 AsyncSuffixCacheClient

异步版本，使用线程池实现非阻塞更新：

```python
class AsyncSuffixCacheClient(SuffixCacheClient):
    """Async version using thread pool for non-blocking updates."""

    def __init__(self, server_urls: list[str], max_pending_updates: int = 100):
        super().__init__(server_urls)
        self._executor = ThreadPoolExecutor(max_workers=8)
        self._pending_futures = []
        self.max_pending_updates = max_pending_updates

    def update_cache_batch_async(self, batch_data: list[dict]):
        """Asynchronously update cache with batch data."""
        self._wait_for_pending_updates(threshold=self.max_pending_updates)
        future = self._executor.submit(self.update_cache_batch, batch_data)
        self._pending_futures.append(future)

    def wait_all(self, timeout: Optional[float] = None):
        """Wait for all pending updates to complete."""
        for future in self._pending_futures:
            future.result(timeout=timeout)
        self._pending_futures.clear()
```

#### 5.4.3 SuffixCacheManager

管理 RL step 后的 cache 同步：

```python
# verl/trainer/ppo/suffix_cache_manager.py

class SuffixCacheManager:
    """Manager for distributed suffix cache synchronization in RL training."""

    def __init__(self, config, actor_rollout_wg, port: int = 6378):
        self.config = config
        self.actor_rollout_wg = actor_rollout_wg
        self.port = port

        # Initialize cache client with server addresses
        self._server_urls = self._get_server_addresses()
        self._cache_client = AsyncSuffixCacheClient(self._server_urls)

    def update_cache(self, batch: DataProto, responses_per_prompt: int = 1):
        """
        Update suffix cache with generated sequences.

        Extracts prompts and responses from batch and broadcasts to all workers.
        """
        batch_data = self._extract_batch_data(batch, responses_per_prompt)
        if batch_data:
            self._cache_client.update_cache_batch_async(batch_data)

    def _extract_batch_data(self, batch: DataProto, responses_per_prompt: int) -> list[dict]:
        """Extract token sequences from batch for cache update."""
        prompts = batch.batch["prompts"]
        responses = batch.batch["responses"]
        attention_mask = batch.batch["attention_mask"]

        response_length = responses.shape[-1]
        prompt_mask = attention_mask[:, :-response_length]
        response_mask = attention_mask[:, -response_length:]

        prompt_lengths = prompt_mask.sum(-1).cpu().tolist()
        response_lengths = response_mask.sum(-1).cpu().tolist()

        batch_data = []
        for i in range(len(prompts)):
            prompt_len = int(prompt_lengths[i])
            response_len = int(response_lengths[i])
            if prompt_len == 0 or response_len == 0:
                continue

            token_ids = prompts[i][:prompt_len].tolist() + responses[i][:response_len].tolist()
            batch_data.append({
                "request_id": str(uuid.uuid4()),
                "token_ids": token_ids,
                "prompt_length": prompt_len,
            })
        return batch_data
```

### 5.5 配置项

在 rollout 配置中添加 suffix cache 配置：

```python
# verl/workers/config/rollout.py

@dataclass
class SuffixCacheConfig(BaseConfig):
    """Configuration for suffix cache synchronization across rollout instances."""

    # 是否启用 suffix cache 同步
    enable: bool = False

    # Suffix cache server 端口
    port: int = 6378

    # 更新请求超时时间（秒）
    update_timeout: float = 5.0

    # 最大并发更新线程数
    max_workers: int = 4

    # 最大待处理更新数量（超过则阻塞）
    max_pending_updates: int = 100

    # 是否阻塞等待更新完成再开始下一步
    blocking_update: bool = False

    # 阻塞等待超时时间
    blocking_timeout: Optional[float] = None


@dataclass
class RolloutConfig(BaseConfig):
    # ... existing fields ...

    # Suffix cache configuration for speculative decoding acceleration
    suffix_cache: SuffixCacheConfig = field(default_factory=SuffixCacheConfig)
```

### 5.6 RayPPOTrainer 集成

在 `RayPPOTrainer` 中集成 suffix cache manager：

```python
# verl/trainer/ppo/ray_trainer.py

class RayPPOTrainer:
    def __init__(self, ...):
        # ... existing initialization ...
        self.suffix_cache_manager = None

    def _init_suffix_cache_manager(self):
        """Initialize suffix cache manager if enabled."""
        suffix_cache_config = self.config.actor_rollout_ref.rollout.get("suffix_cache", {})
        if not suffix_cache_config.get("enable", False):
            return

        from verl.trainer.ppo.suffix_cache_manager import SuffixCacheManager
        self.suffix_cache_manager = SuffixCacheManager(
            config=self.config,
            actor_rollout_wg=self.actor_rollout_wg,
            port=suffix_cache_config.get("port", 6378),
        )

    def init_workers(self):
        # ... existing worker initialization ...
        self._init_suffix_cache_manager()

    def fit(self):
        for batch_dict in self.train_dataloader:
            # ... generate sequences ...
            batch = batch.union(gen_batch_output)

            # Update suffix cache with generated sequences
            if self.suffix_cache_manager is not None and self.suffix_cache_manager.enabled:
                self.suffix_cache_manager.update_cache(
                    batch,
                    self.config.actor_rollout_ref.rollout.n
                )

            # ... rest of training loop ...
```

### 5.7 使用方式

在训练配置中启用 suffix cache 同步：

```yaml
actor_rollout_ref:
  rollout:
    name: sglang  # 必须使用 sglang rollout
    suffix_cache:
      enable: true
      port: 6378
      max_workers: 4
      max_pending_updates: 100
```

启动 sglang server 时需要启用 suffix cache server：

```bash
python -m sglang.launch_server \
    --model-path /path/to/model \
    --speculative-algorithm SUFFIX \
    --speculative-suffix-cache-server-port 6378 \
    --speculative-disable-batch-size-threshold 16
```

### 5.8 性能影响

| 场景 | 无同步 | 有同步 | 说明 |
|------|--------|--------|------|
| 网络延迟 | 0 | ~5-10ms/step | 异步更新，不阻塞训练 |
| 内存开销 | 0 | ~10MB | 缓存待发送的 batch 数据 |
| Cache 命中率提升 | - | 2-5x | 取决于 prompt 相似度 |
| 接受率提升 | baseline | +10-20% | 更丰富的历史模式 |

---

## 6. 未来优化方向


### 6.1 动态 Suffix Decoding 配置

当前实现仅支持二值开关（启用/禁用），可扩展为多级配置，根据 batch size 动态选择最优策略。

#### 配置预设

```python
@dataclass
class SpeculationConfig:
    """Multi-level speculation configuration."""

    # 关闭模式：batch_size > disable_threshold
    disable_threshold: int = 32

    # 保守模式：conservative_threshold < batch_size <= disable_threshold
    conservative_threshold: int = 16
    conservative_spec_length: int = 16
    conservative_max_candidates: int = 4

    # 激进模式：batch_size <= conservative_threshold
    aggressive_spec_length: int = 64
    aggressive_max_candidates: int = 16
    aggressive_tree_depth: int = 16
```

#### 策略选择逻辑

```python
def get_speculation_mode(batch_size: int, config: SpeculationConfig) -> str:
    if batch_size > config.disable_threshold:
        return "disabled"
    elif batch_size > config.conservative_threshold:
        return "conservative"
    else:
        return "aggressive"
```

#### 参数对比

| 参数 | 关闭 | 保守 | 激进 |
|------|------|------|------|
| 适用 batch_size | > 32 | 16-32 | < 16 |
| speculation_length | 0 | 16 | 64-128 |
| max_candidates | 0 | 4 | 16 |
| suffix_tree_depth | - | 8 | 16 |
| 适用场景 | 高负载 | 中等负载 | 低负载/长尾 |

#### 实现要点

1. **CUDA Graph 多组捕获：** 为每种模式分别捕获 graph
2. **平滑切换：** 确保 mode 切换时的状态一致性
3. **动态阈值：** 可根据实时接受率调整阈值


### 6.2 动态负载感知调度

当前调度策略使用最少连接（min-heap）负载均衡 + Sticky Session，但由于 generation 长度不可预测，可能出现严重负载不均衡。特别是在长尾场景下，大部分请求已完成，少数请求仍在运行，导致大量算力闲置。

#### 优化方案

通过 Time-slicing 调度 + KV Cache 迁移实现动态负载均衡：

```
┌─────────────────────────────────────────────────────────────────┐
│                    长尾场景的算力浪费                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  正常状态:                                                       │
│  Server 0: [req1, req2, req3, req4]  batch_size=4               │
│  Server 1: [req5, req6, req7, req8]  batch_size=4               │
│  Server 2: [req9, req10, req11, req12] batch_size=4             │
│                                                                  │
│  长尾状态（大部分请求完成）:                                     │
│  Server 0: [长尾_req_A]  batch_size=1                           │
│  Server 1: [idle]        batch_size=0                           │
│  Server 2: [idle]        batch_size=0                           │
│                                                                  │
│  问题: Server 1, 2 完全闲置，对长尾加速没有任何贡献              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

#### Phase 1: 正常生成

```
┌─────────────────────────────────────────────────────────────────┐
│                         正常阶段                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  • 标准调度，正常 batch size                                    │
│  • 保守的 suffix decoding 参数                                  │
│  • 每个请求设置固定 max_tokens 切片（如 8192）                  │
│  • Time-slicing 后可重新调度                                    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

#### Phase 2: 长尾检测

```
┌─────────────────────────────────────────────────────────────────┐
│                         检测阶段                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  监控指标:                                                       │
│  • idle_servers / total_servers > threshold                     │
│  • running_requests < low_watermark                             │
│  • 平均 batch_size < target_batch_size                          │
│                                                                  │
│  触发条件: 满足任一条件即进入长尾优化模式                       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

#### Phase 3: 动态重调度

```
┌─────────────────────────────────────────────────────────────────┐
│                         重调度阶段                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  操作:                                                           │
│  1. 将长尾请求分散到多个 idle server                            │
│  2. KV Cache 迁移到目标 server（利用分布式 KV Cache）           │
│  3. 调整 suffix decoding 参数为激进模式:                        │
│     - speculation_length: 32 → 128                              │
│     - max_candidates: 4 → 16                                    │
│     - suffix_tree_depth: 8 → 16                                 │
│                                                                  │
│  效果:                                                           │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ Server 0: [长尾_req_A] batch_size=1, speculation_window=64│ │
│  │ Server 1: [长尾_req_A] batch_size=1, speculation_window=64│ │
│  │ Server 2: [长尾_req_A] batch_size=1, speculation_window=64│ │
│  │                 ↑                                          │ │
│  │         同一请求复制到多个 server                           │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  每个 instance:                                                  │
│  • batch_size 小 → KV Cache 空间充足                            │
│  • 激进 speculation → 更长的 draft chain                        │
│  • 空闲算力充分利用                                              │
│  • 长尾请求加速明显                                              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 7. 总结

| 模块 | 当前实现 | 未来优化 |
|------|----------|----------|
| 启停策略 | 静态阈值 | MAB 动态决策 |
| Draft Tokens | 固定数量 | 根据接受率动态调整 |
| CUDA Graph | 两组 graphs | 多组 graphs 支持多参数 |
| 外部接口 | HTTP Server | gRPC + 批量更新 |
| verl 集成 | 异步广播同步 | 智能选择性同步 |
