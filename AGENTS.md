# verl Project Context

## Project Overview

**verl** (Volcano Engine Reinforcement Learning for LLMs) is a flexible, efficient, and production-ready RL training library for large language models (LLMs). It is the open-source version of the [HybridFlow](https://arxiv.org/abs/2409.19256v2) paper.

### Key Features

- **Training Backends**: FSDP, FSDP2, and Megatron-LM
- **Rollout Engines**: vLLM, SGLang, and HuggingFace Transformers
- **RL Algorithms**: PPO, GRPO, GSPO, ReMax, REINFORCE++, RLOO, PRIME, DAPO, DrGRPO, etc.
- **Model Support**: Qwen, Llama, Gemma, DeepSeek, and other HuggingFace models
- **Hardware**: NVIDIA GPU, AMD GPU (ROCm), Ascend NPU

## Project Structure

```
verl/
├── verl/                    # Core source code
│   ├── trainer/             # Training logic (PPO, SFT, etc.)
│   │   ├── config/          # Hydra configuration files (YAML)
│   │   ├── ppo/             # PPO algorithm implementation
│   │   └── main_ppo.py      # PPO training entry point
│   ├── workers/             # Distributed workers
│   │   ├── actor/           # Actor workers
│   │   ├── critic/          # Critic workers
│   │   ├── rollout/         # Rollout generation
│   │   └── reward_model/    # Reward model workers
│   ├── models/              # Model implementations
│   ├── utils/               # Utility functions
│   ├── protocol.py          # DataProto - data transfer protocol
│   └── base_config.py       # Base configuration class
├── examples/                # Example training scripts
│   ├── ppo_trainer/         # PPO examples
│   ├── grpo_trainer/        # GRPO examples
│   ├── sft/                 # Supervised fine-tuning examples
│   └── sglang_multiturn/    # Multi-turn RL examples
├── tests/                   # Test suites
├── docs/                    # Documentation
├── scripts/                 # Utility scripts
└── recipe/                  # Training recipes (git submodule)
```

## Building and Running

### Installation

```bash
# Basic installation with vLLM
pip install -e .[test,vllm]

# Or with SGLang
pip install -e .[test,sglang]

# Full dependencies for development
pip install -r requirements.txt
```

### Running Training

PPO training example:
```bash
python3 -m verl.trainer.main_ppo \
    data.train_files="['/path/to/train.parquet']" \
    data.val_files="['/path/to/val.parquet']" \
    actor_rollout_ref.model.path="/path/to/model" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1
```

SFT training example:
```bash
python3 -m verl.trainer.sft_trainer \
    data.train_files="/path/to/train.parquet" \
    model.path="/path/to/model"
```

### Testing

```bash
# Run specific test
pytest tests/test_protocol_on_cpu.py

# Run tests with specific marker
pytest tests/ -m "not gpu"
```

### Code Quality

```bash
# Setup pre-commit hooks
pip install pre-commit
pre-commit install

# Run linting
pre-commit run --all-files

# Run ruff specifically
pre-commit run --all-files --show-diff-on-failure ruff

# Type checking (for specific modules)
mypy verl/trainer/ppo/core_algos.py
```

### Building Documentation

```bash
cd docs
pip install -r requirements-docs.txt
make clean
make html
python -m http.server -d _build/html/
```

## Configuration System

verl uses **Hydra** for configuration management. Configuration files are in `verl/trainer/config/`:

- `ppo_trainer.yaml` - Default PPO configuration
- `ppo_megatron_trainer.yaml` - Megatron-LM backend configuration
- `sft_trainer.yaml` - SFT configuration

### Key Configuration Sections

- `actor_rollout_ref` - Actor, rollout, and reference model config
- `critic` - Critic model config
- `reward_model` - Reward model config
- `algorithm` - Algorithm hyperparameters (gamma, lam, adv_estimator, etc.)
- `trainer` - Training settings (epochs, logging, checkpointing)
- `data` - Data loading configuration

### Overriding Config

```bash
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gae \
    trainer.total_epochs=15 \
    actor_rollout_ref.model.path="/path/to/model"
```

## Development Conventions

### Code Style

- **Line Length**: 120 characters (configured in pyproject.toml)
- **Linter**: ruff (pycodestyle, Pyflakes, pyupgrade, flake8-bugbear, isort)
- **Formatter**: ruff format
- **Type Hints**: mypy (strict for specific modules)

### Import Conventions

```python
# Known first-party imports (configured in ruff)
from verl.protocol import DataProto
from verl.utils.torch_functional import allgather_dict_tensors
```

### Data Protocol

`DataProto` is the core data transfer class between modules:
- Located in `verl/protocol.py`
- Wraps TensorDict for batch data
- Supports padding, concatenation, and distributed operations

### Testing Conventions

- CPU tests: `tests/test_*_on_cpu.py`
- GPU tests: `tests/trainer/`, `tests/workers/`
- E2E tests: `tests/special_e2e/`
- Tests run on GitHub Actions (see `.github/workflows/`)

## Key Modules

### `verl.trainer.ppo`

- `ray_trainer.py` - Ray-based distributed PPO trainer
- `core_algos.py` - Core PPO algorithms (GAE, advantage computation)
- `reward/` - Reward function implementations

### `verl.workers`

- `fsdp_workers.py` - FSDP-based workers
- `megatron_workers.py` - Megatron-LM workers
- `actor/` - Actor implementation
- `critic/` - Critic implementation
- `rollout/` - Rollout generation (vLLM, SGLang)

### `verl.models`

- Model registration and utilities
- Transformers model patches
- NPU-specific patches

## Common Tasks

### Adding a New Model

1. Ensure HuggingFace compatibility
2. Add model config in `verl/trainer/config/model/`
3. Test with FSDP backend first
4. For Megatron backend, see `docs/advance/megatron_extension.md`

### Adding a New RL Algorithm

1. Implement advantage estimator in `verl/trainer/ppo/core_algos.py`
2. Add algorithm config in `verl/trainer/config/algorithm/`
3. Create example script in `examples/<algo>_trainer/`

### Adding a Custom Reward Function

1. Create a Python file with a `compute_score` function
2. Pass the path via config:
```bash
custom_reward_function.path="/path/to/reward.py" \
custom_reward_function.name="compute_score"
```

## Important Notes

### Environment Variables

- `VERL_USE_EXTERNAL_MODULES` - Load external modules
- `VERL_USE_MODELSCOPE` - Use ModelScope hub instead of HuggingFace
- `VERL_AUTO_PADDING` - Enable auto-padding for DataProto

### Git Submodules

The `recipe/` directory is a git submodule. Initialize it with:
```bash
git submodule update --init --recursive recipe
```

### Breaking Changes

Check `.github/workflows/` and `docs/` for version-specific notes and upgrade guides.

## Resources

- [Documentation](https://verl.readthedocs.io/en/latest/)
- [GitHub Issues](https://github.com/volcengine/verl/issues)
- [Slack](https://join.slack.com/t/verl-project/)
- [Paper](https://arxiv.org/abs/2409.19256)
