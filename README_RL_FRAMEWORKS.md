# RL Training Frameworks: RL-Games vs RSL-RL

This document explains how **RL-Games** and **RSL-RL** work in this codebase, their differences, and when to use each framework.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [RL-Games Framework](#rl-games-framework)
- [RSL-RL Framework](#rsl-rl-framework)
- [Key Differences](#key-differences)
- [Usage Examples](#usage-examples)
- [When to Use Which](#when-to-use-which)

---

## Overview

Both **RL-Games** and **RSL-RL** are reinforcement learning frameworks that integrate with Isaac Sim to train RL agents. They follow similar workflows but differ in their configuration style, API design, and setup complexity.

### Quick Comparison

| Aspect | RL-Games | RSL-RL |
|--------|----------|--------|
| **Config Format** | YAML files | Python dataclasses |
| **Type Safety** | Runtime (dictionary) | Compile-time (typed) |
| **Setup Complexity** | More setup required | Simpler API |
| **Environment Registration** | Registry-based | Direct wrapper |
| **Best For** | Multiple algorithms, YAML preference | Type safety, simpler code |

---

## Prerequisites

Before using either RL-Games or RSL-RL frameworks, ensure you have the proper environment setup:

### Conda Environment

This project uses the **`env_isaaclab`** conda environment. Make sure it's activated before running any training scripts:

```bash
conda activate env_isaaclab
```

### Required Dependencies

Install the required Python packages, including TensorBoard for logging and visualization:

```bash
# Install TensorBoard (required for training logs and visualization)
# 2.20.0
pip install tensorboard 

# Or if using Isaac Lab's environment manager
./isaaclab.sh -p -m pip install tensorboard 
```

**Note**: TensorBoard is required for viewing training metrics and is used by both RL-Games and RSL-RL frameworks for logging. If you encounter a `ModuleNotFoundError: No module named 'tensorboard'`, install it using the command above.

### Verify Installation

You can verify TensorBoard is installed correctly:

```bash
python -c "import tensorboard; print(tensorboard.__version__)"
```

---

## RL-Games Framework

### What is RL-Games?

RL-Games is a popular RL framework that uses **YAML-based configuration** and a **registry system** for environment management.

### Architecture

#### 1. **Configuration System**
- Uses YAML files (e.g., `rl_games_ppo_cfg.yaml`)
- Nested dictionary structure under `params` key
- Configuration includes:
  - Algorithm settings (`algo`)
  - Model architecture (`model`, `network`)
  - Training hyperparameters (`config`)
  - Environment wrapper settings (`env`)

#### 2. **Environment Registration**
RL-Games requires registering your Isaac Lab environment in its internal registry:

```python
# Register environment wrapper
vecenv.register(
    "IsaacRlgWrapper", 
    lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
)

# Register environment configuration
env_configurations.register(
    "rlgpu",  # Must use "rlgpu" as env_name in config
    {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env}
)
```

#### 3. **Training Flow**

```python
# 1. Create runner with observer
runner = Runner(IsaacAlgoObserver())

# 2. Load configuration from YAML
runner.load(agent_cfg)

# 3. Reset and train
runner.reset()
runner.run({"train": True, "play": False})
```

### Configuration Example

```yaml
params:
  seed: 42
  
  # Environment wrapper settings
  env:
    clip_observations: 5.0
    clip_actions: 1.0
  
  # Algorithm selection
  algo:
    name: a2c_continuous
  
  # Model architecture
  model:
    name: continuous_a2c_logstd
  
  # Network configuration
  network:
    name: actor_critic
    mlp:
      units: [32, 32]
      activation: elu
  
  # Training configuration
  config:
    name: cartpole_direct
    env_name: rlgpu  # Must be "rlgpu"
    device: 'cuda:0'
    num_actors: -1  # Auto-set from num_envs
    max_epochs: 150
    learning_rate: 5e-4
    gamma: 0.99
    horizon_length: 32
    minibatch_size: 32
    mini_epochs: 8
    # ... more PPO hyperparameters
```

### Key Features

- ✅ **Multiple Algorithms**: Supports PPO, A2C, and other algorithms via config
- ✅ **Flexible Configuration**: Easy to modify hyperparameters in YAML
- ✅ **Custom Observer**: `IsaacAlgoObserver` for Isaac Sim integration
- ⚠️ **Registry Required**: Must register environment as `"rlgpu"`
- ⚠️ **Dictionary-based**: No type checking at development time

---

## RSL-RL Framework

### What is RSL-RL?

RSL-RL (Rapid Sim-to-Real Learning) is a framework designed for Isaac Sim with **Python-based configuration** and a **simpler, more direct API**.

### Architecture

#### 1. **Configuration System**
- Uses Python dataclasses/ConfigClass
- Type-safe configuration with IDE autocomplete
- Defined in Python files (e.g., `rsl_rl_ppo_cfg.py`)

#### 2. **Direct Environment Wrapping**
No registry needed - directly wrap the environment:

```python
env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
```

#### 3. **Training Flow**

```python
# 1. Create runner directly
runner = OnPolicyRunner(
    env, 
    agent_cfg.to_dict(), 
    log_dir=log_dir, 
    device=agent_cfg.device
)

# 2. Optional: Add git info to logs
runner.add_git_repo_to_log(__file__)

# 3. Load checkpoint if resuming
if agent_cfg.resume:
    runner.load(resume_path)

# 4. Train
runner.learn(num_learning_iterations=agent_cfg.max_iterations)
```

### Configuration Example

```python
@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Configuration for PPO agent."""
    
    # Experiment settings
    experiment_name: str = "dexhand_single"
    run_name: str = ""
    max_iterations: int = 150
    seed: int = 42
    
    # Device and environment
    device: str = "cuda:0"
    clip_actions: float = 1.0
    
    # Algorithm configuration
    algorithm: PPOAlgorithmCfg = PPOAlgorithmCfg(
        name="ppo",
        value_loss_coef=2.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,
        num_learning_epochs=8,
        num_mini_batches=32,
        # ... more settings
    )
    
    # Policy network
    policy: ActorCriticCfg = ActorCriticCfg(...)
```

### Key Features

- ✅ **Type Safety**: Compile-time type checking with dataclasses
- ✅ **Simple API**: Direct runner creation, no registry needed
- ✅ **IDE Support**: Autocomplete and type hints
- ✅ **Built-in Logging**: Git state tracking included
- ✅ **Cleaner Code**: More Pythonic approach

---

## Key Differences

### 1. **Configuration Style**

**RL-Games (YAML):**
```yaml
params:
  config:
    learning_rate: 5e-4
    max_epochs: 150
```

**RSL-RL (Python):**
```python
@configclass
class Config:
    learning_rate: float = 5e-4
    max_iterations: int = 150
```

### 2. **Environment Setup**

**RL-Games:**
```python
# Requires registration
env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions)
env_configurations.register("rlgpu", {"env_creator": lambda **kwargs: env})
```

**RSL-RL:**
```python
# Direct wrapping
env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
```

### 3. **Runner Creation**

**RL-Games:**
```python
runner = Runner(IsaacAlgoObserver())
runner.load(agent_cfg)  # Load from dict
```

**RSL-RL:**
```python
runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir)
```

### 4. **Training Method**

**RL-Games:**
```python
runner.run({"train": True, "play": False})
```

**RSL-RL:**
```python
runner.learn(num_learning_iterations=agent_cfg.max_iterations)
```

---

## Usage Examples

### Running RL-Games Training

```bash
# Basic training
python scripts/rl_games/train.py \
    --task Isaac-Cartpole-Direct-v0 \
    --num_envs 4096

# With custom checkpoint
python scripts/rl_games/train.py \
    --task Isaac-Cartpole-Direct-v0 \
    --checkpoint logs/rl_games/cartpole_direct/2024-01-01_12-00-00/nn/cartpole_direct.pth

# With video recording
python scripts/rl_games/train.py \
    --task Isaac-Cartpole-Direct-v0 \
    --video \
    --video_interval 2000
```

### Running RSL-RL Training

```bash
# Basic training
python scripts/rsl_rl/train.py \
    --task Isaac-Cartpole-Direct-v0 \
    --num_envs 4096

# Resume training
python scripts/rsl_rl/train.py \
    --task Isaac-Cartpole-Direct-v0 \
    --resume \
    --load_run 2024-01-01_12-00-00

# With custom iterations
python scripts/rsl_rl/train.py \
    --task Isaac-Cartpole-Direct-v0 \
    --max_iterations 500
```

### Complete Code Example: RL-Games

```python
# 1. Launch Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# 2. Load configurations (via Hydra)
@hydra_task_config(args_cli.task, "rl_games_cfg_entry_point")
def main(env_cfg, agent_cfg):
    # 3. Create environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    
    # 4. Wrap for RL-Games
    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions)
    
    # 5. Register environment
    env_configurations.register("rlgpu", {"env_creator": lambda **kwargs: env})
    
    # 6. Create and configure runner
    runner = Runner(IsaacAlgoObserver())
    runner.load(agent_cfg)
    
    # 7. Train
    runner.run({"train": True})
```

### Complete Code Example: RSL-RL

```python
# 1. Launch Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# 2. Load configurations (via Hydra)
@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    # 3. Create environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    
    # 4. Wrap for RSL-RL
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    
    # 5. Create runner
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir)
    
    # 6. Train
    runner.learn(num_learning_iterations=agent_cfg.max_iterations)
```

---

## When to Use Which

### Use **RL-Games** if:

- ✅ You prefer **YAML configuration** files
- ✅ You need to **switch between multiple algorithms** easily
- ✅ You're **migrating from existing RL-Games projects**
- ✅ You want **flexible hyperparameter tuning** via config files
- ✅ You're comfortable with **dictionary-based configurations**

### Use **RSL-RL** if:

- ✅ You prefer **Python-based, type-safe configurations**
- ✅ You want **IDE autocomplete and type checking**
- ✅ You prefer a **simpler, more direct API**
- ✅ You want **cleaner, more maintainable code**
- ✅ You're starting a **new project** from scratch

---

## File Structure

### RL-Games Files

```
scripts/rl_games/
├── train.py          # Training script
├── play.py           # Evaluation script
└── ...

source/Dexhand_single/.../agents/
└── rl_games_ppo_cfg.yaml  # Agent configuration
```

### RSL-RL Files

```
scripts/rsl_rl/
├── train.py          # Training script
├── play.py           # Evaluation script
├── cli_args.py       # CLI argument handling
└── ...

source/Dexhand_single/.../agents/
└── rsl_rl_ppo_cfg.py  # Agent configuration (Python)
```

---

## Logging

Both frameworks save logs to similar directory structures:

### RL-Games Logs
```
logs/rl_games/
└── {experiment_name}/
    └── {timestamp}/
        ├── params/
        │   ├── env.yaml
        │   ├── agent.yaml
        │   ├── env.pkl
        │   └── agent.pkl
        ├── nn/
        │   └── {checkpoint}.pth
        └── summaries/
```

### RSL-RL Logs
```
logs/rsl_rl/
└── {experiment_name}/
    └── {timestamp}_{run_name}/
        ├── params/
        │   ├── env.yaml
        │   ├── agent.yaml
        │   ├── env.pkl
        │   └── agent.pkl
        ├── model_{iteration}.pt
        └── progress.csv
```

---

## Summary

Both **RL-Games** and **RSL-RL** are powerful frameworks for training RL agents in Isaac Sim. The main differences are:

- **RL-Games**: YAML configs, registry-based, more flexible for multiple algorithms
- **RSL-RL**: Python configs, direct API, type-safe, simpler code

Choose based on your preference for configuration style and API complexity. Both work seamlessly with Isaac Lab environments!

---

## Additional Resources

- [RL-Games Documentation](https://github.com/Denys88/rl_games)
- [RSL-RL Documentation](https://github.com/leggedrobotics/rsl_rl)
- [Isaac Lab Documentation](https://isaac-sim.github.io/IsaacLab/)

