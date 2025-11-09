# RL Training Frameworks: RL-Games vs RSL-RL

This document explains how **RL-Games** and **RSL-RL** work in this codebase, their differences, and when to use each framework.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [How RL Training Works](#how-rl-training-works)
- [RL-Games Framework](#rl-games-framework)
- [RSL-RL Framework](#rsl-rl-framework)
- [Key Differences](#key-differences)
- [Quick Start Guide](#quick-start-guide)
- [Dexhand Single Training](#dexhand-single-training)
- [Recording Videos](#recording-videos)
- [Additional Resources](#additional-resources)

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

## How RL Training Works

This section explains how reinforcement learning training works in this codebase, from the basic concepts to the actual implementation.

### What is Reinforcement Learning?

Reinforcement Learning (RL) is a type of machine learning where an **agent** learns to make decisions by interacting with an **environment**. The agent receives:
- **Observations**: Information about the current state (e.g., joint positions, velocities)
- **Rewards**: Feedback on how good/bad its actions are
- **Actions**: Commands it can execute (e.g., motor torques)

The goal is to learn a **policy** (a neural network) that maximizes cumulative rewards over time.

### The RL Training Loop

Here's how training works step-by-step:

```
┌─────────────────────────────────────────────────────────────┐
│                    RL Training Loop                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. Initialize Environment                                   │
│     └─> Create N parallel environments (e.g., 4096)        │
│                                                              │
│  2. Reset Environments                                      │
│     └─> Get initial observations (obs)                      │
│                                                              │
│  3. Agent Decides Actions                                    │
│     └─> Policy network: obs → actions                       │
│                                                              │
│  4. Execute Actions in Environment                           │
│     └─> Step simulation forward                             │
│     └─> Get new observations, rewards, done flags          │
│                                                              │
│  5. Store Experience                                        │
│     └─> Save (obs, action, reward, next_obs, done)         │
│                                                              │
│  6. Learn from Experience (Periodically)                     │
│     └─> Update policy network using collected data          │
│     └─> Improve future decisions                             │
│                                                              │
│  7. Repeat from Step 2                                       │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Key Components in This Codebase

#### 1. **Environment (Isaac Sim)**

The environment is a physics simulation running in Isaac Sim:

```python
# Create environment with N parallel instances
env = gym.make("Isaac-Cartpole-Direct-v0", cfg=env_cfg)
# num_envs = 4096 means 4096 parallel simulations!
```

**Why parallel environments?**
- **Speed**: Run thousands of simulations simultaneously on GPU
- **Diversity**: Each environment can have different initial conditions
- **Efficiency**: Collect more experience in less time

**What the environment does:**
- Simulates physics (cart moving, pole balancing)
- Provides observations (cart position, pole angle, velocities)
- Computes rewards (positive for balancing, negative for falling)
- Returns done flags (episode finished when pole falls)

#### 2. **Agent (Neural Network Policy)**

The agent is a neural network that maps observations to actions:

```python
# Policy network architecture
observations → [Neural Network] → actions
     ↓                              ↓
  (state info)              (motor commands)
```

**The policy learns:**
- **What to do** in each situation
- **How to maximize rewards** over time
- **General strategies** that work across different scenarios

#### 3. **Training Process**

Here's what happens when you run `train.py`:

```python
# 1. Setup
env = gym.make(task, cfg=env_cfg)  # Create environment
runner = Runner()                  # Create RL trainer
runner.load(agent_cfg)             # Load algorithm config

# 2. Training loop (happens inside runner.run())
for epoch in range(max_epochs):
    # Collect experience
    for step in range(horizon_length):
        obs = env.get_observations()
        actions = policy(obs)              # Agent decides
        obs, rewards, dones = env.step(actions)  # Execute
        
        # Store experience
        buffer.store(obs, actions, rewards, ...)
    
    # Learn from experience
    for mini_epoch in range(mini_epochs):
        batch = buffer.sample(minibatch_size)
        loss = compute_loss(batch)
        optimizer.step(loss)  # Update neural network
```

### How the Agent Learns

The agent uses **Policy Gradient** methods (like PPO - Proximal Policy Optimization):

1. **Collect Experience**: Run many episodes, storing (observation, action, reward)
2. **Compute Advantages**: Estimate how good each action was compared to average
3. **Update Policy**: Adjust neural network weights to:
   - Increase probability of good actions
   - Decrease probability of bad actions
4. **Repeat**: Keep collecting new experience and improving

**Example (Cartpole):**
- **Bad action**: Pole falls → negative reward → Policy learns to avoid this
- **Good action**: Pole stays balanced → positive reward → Policy learns to repeat this

### Parallel Environments: The Key to Speed

Isaac Sim's power comes from running **thousands of environments in parallel**:

```
Traditional RL:          Isaac Sim RL:
┌─────────┐             ┌─────┬─────┬─────┬─────┐
│ Env 1   │             │Env 1│Env 2│Env 3│ ... │
│ (CPU)   │             │     │     │     │4096 │
└─────────┘             └─────┴─────┴─────┴─────┘
     ↓                        ↓ (all on GPU)
  Slow!                   Super Fast!
```

**Benefits:**
- **4096 environments** = 4096x more data per second
- All run on **GPU** simultaneously
- Training that would take days now takes hours

### The Complete Training Workflow

When you run:
```bash
python scripts/rl_games/train.py --task Isaac-Cartpole-Direct-v0 --num_envs 4096
```

Here's what happens:

1. **Initialization** (one time):
   - Launch Isaac Sim
   - Create 4096 parallel cartpole environments
   - Initialize neural network (random weights)
   - Set up optimizer, buffers, etc.

2. **Training Loop** (repeats for many epochs):
   ```
   Epoch 1:
   - Run 4096 environments for 32 steps each
   - Collect ~131,000 (obs, action, reward) tuples
   - Update policy network 8 times using this data
   - Save checkpoint
   
   Epoch 2:
   - Agent is slightly better now
   - Run environments again, collect new data
   - Update policy again
   - ...
   
   Epoch 150:
   - Agent has learned to balance the pole!
   - Training complete
   ```

3. **Learning Progress**:
   - Early epochs: Agent is random, pole falls quickly
   - Mid epochs: Agent starts learning, pole stays up longer
   - Late epochs: Agent masters the task, consistently balances

### Key Concepts

#### **Observations**
What the agent "sees":
- Cart position, velocity
- Pole angle, angular velocity
- Any other relevant state information

#### **Actions**
What the agent "does":
- Motor torques/forces to apply
- Discrete choices (left/right) or continuous values

#### **Rewards**
Feedback signal:
- **Positive**: Good behavior (pole balanced)
- **Negative**: Bad behavior (pole fell)
- **Zero**: Neutral state

#### **Episodes**
A complete trial:
- Start: Reset environment to initial state
- Middle: Agent acts, receives rewards
- End: Episode terminates (pole falls, time limit, etc.)

### Why This Approach Works

1. **Trial and Error**: Agent tries many actions, learns which work
2. **Reward Signal**: Clear feedback on what's good/bad
3. **Parallel Exploration**: Thousands of environments = diverse experience
4. **Neural Network**: Can learn complex patterns and generalize
5. **Iterative Improvement**: Each epoch makes the agent slightly better

### Example: Cartpole Training

**Goal**: Balance a pole on a cart

**Observations**: Cart position, pole angle, velocities  
**Actions**: Force to apply to cart (left/right)  
**Reward**: +1 for each step pole stays balanced, -100 if pole falls

**Training Process**:
1. **Epoch 1**: Random actions → Pole falls immediately → Average reward: ~20
2. **Epoch 50**: Agent learning → Pole stays up longer → Average reward: ~150
3. **Epoch 150**: Agent mastered → Pole balanced consistently → Average reward: ~280

The neural network learns the relationship: "If pole tilts right, push cart right" through thousands of trials.

### Summary

RL training in this codebase:
1. Creates **parallel environments** in Isaac Sim (GPU-accelerated)
2. Uses a **neural network** to decide actions
3. Collects **experience** (observations, actions, rewards)
4. Updates the **policy** to maximize rewards
5. Repeats until the agent learns the task

The combination of **Isaac Sim's parallel simulation** + **RL algorithms** + **GPU acceleration** makes training very fast compared to traditional RL methods.

---

## Prerequisites

### Conda Environment

Activate the required conda environment:

```bash
conda activate env_isaaclab
```

### Required Dependencies

Install TensorBoard for logging and visualization:

```bash
# Install TensorBoard (version 2.20.0)
pip install tensorboard

# Or using Isaac Lab's environment manager
./isaaclab.sh -p -m pip install tensorboard
```

**Note**: If you encounter `ModuleNotFoundError: No module named 'tensorboard'`, install it using the command above.

### Verify Installation

```bash
python -c "import tensorboard; print(tensorboard.__version__)"
```

---

## How RL Training Works

### Core Concept

Reinforcement Learning (RL) trains an **agent** (neural network) to make decisions by interacting with an **environment**:

- **Observations**: Current state (e.g., joint positions, velocities)
- **Actions**: Commands to execute (e.g., motor torques)
- **Rewards**: Feedback signal (positive for good behavior, negative for bad)

**Goal**: Learn a policy that maximizes cumulative rewards.

### Training Loop

```
1. Initialize → Create N parallel environments (e.g., 4096)
2. Reset → Get initial observations
3. Agent decides → Policy network: obs → actions
4. Execute → Step simulation, get rewards
5. Store → Save (obs, action, reward) tuples
6. Learn → Update policy network periodically
7. Repeat → Continue from step 2
```

### Why Parallel Environments?

Isaac Sim runs **thousands of environments simultaneously on GPU**:

- **4096 environments** = 4096x more data per second
- Training that would take days now takes hours
- Each environment can have different initial conditions for diversity

### Learning Process

The agent uses **Policy Gradient** methods (PPO):

1. **Collect Experience**: Run episodes, store (observation, action, reward)
2. **Compute Advantages**: Estimate action quality vs. average
3. **Update Policy**: Adjust neural network to increase good actions, decrease bad ones
4. **Repeat**: Continue collecting and improving

**Example**: Cartpole agent learns "if pole tilts right, push cart right" through trial and error.

---

## RL-Games Framework

### Overview

RL-Games uses **YAML-based configuration** and a **registry system** for environment management.

### Key Components

1. **Configuration**: YAML file with nested `params` structure
2. **Environment Registration**: Must register as `"rlgpu"` in RL-Games registry
3. **Training**: Uses `Runner` class with `IsaacAlgoObserver`

### Configuration Example

```yaml
params:
  seed: 42
  env:
    clip_observations: 5.0
    clip_actions: 1.0
  algo:
    name: a2c_continuous
  config:
    name: dexhand_single
    env_name: rlgpu  # Must be "rlgpu"
    device: 'cuda:0'
    max_epochs: 150
    learning_rate: 5e-4
    # ... more hyperparameters
```

### Training Flow

```python
# 1. Create runner
runner = Runner(IsaacAlgoObserver())

# 2. Load YAML config
runner.load(agent_cfg)

# 3. Train
runner.reset()
runner.run({"train": True, "play": False})
```

### Features

- ✅ Multiple algorithms (PPO, A2C, etc.) via config
- ✅ Flexible YAML configuration
- ⚠️ Requires environment registry setup
- ⚠️ No compile-time type checking

---

## RSL-RL Framework

### Overview

RSL-RL (Rapid Sim-to-Real Learning) uses **Python-based configuration** with a **simpler, direct API**.

### Key Components

1. **Configuration**: Python dataclass/ConfigClass (type-safe)
2. **Environment Wrapping**: Direct wrapper, no registry needed
3. **Training**: Uses `OnPolicyRunner` class

### Configuration Example

```python
@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    experiment_name: str = "dexhand_single"
    max_iterations: int = 150
    seed: int = 42
    device: str = "cuda:0"
    algorithm: PPOAlgorithmCfg = PPOAlgorithmCfg(...)
    policy: ActorCriticCfg = ActorCriticCfg(...)
```

### Training Flow

```python
# 1. Create runner directly
runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir)

# 2. Train
runner.learn(num_learning_iterations=agent_cfg.max_iterations)
```

### Features

- ✅ Type-safe configuration (IDE autocomplete)
- ✅ Simpler API (no registry)
- ✅ Built-in git state tracking
- ✅ More Pythonic code

---

## Key Differences

| Feature | RL-Games | RSL-RL |
|---------|----------|--------|
| **Config** | YAML (dictionary) | Python (dataclass) |
| **Environment Setup** | Registry required | Direct wrapper |
| **Runner Creation** | `Runner().load(config)` | `OnPolicyRunner(env, config)` |
| **Training Method** | `runner.run()` | `runner.learn()` |
| **Type Safety** | Runtime | Compile-time |

---

## Quick Start Guide

### Finding Your Checkpoint Paths

**Step 1: List experiments**
```bash
ls logs/rl_games/  # or logs/rsl_rl/
```

**Step 2: List training runs** (timestamp directories = log directories)
```bash
ls logs/rl_games/{experiment_name}/
# Example: 2025-11-08_15-30-10
```

**Step 3: Check available checkpoints**
```bash
ls logs/rl_games/{experiment_name}/{timestamp}/nn/
```

**Quick tip**: Get most recent checkpoint
```bash
ls -t logs/rl_games/{experiment_name}/*/nn/*.pth | head -1
```

### Training Commands

#### RL-Games

```bash
# Basic training
python scripts/rl_games/train.py \
    --task Isaac-Cartpole-Direct-v0 \
    --num_envs 4096

# Resume from checkpoint (creates new log directory)
python scripts/rl_games/train.py \
    --task Isaac-Cartpole-Direct-v0 \
    --checkpoint logs/rl_games/cartpole_direct/{timestamp}/nn/cartpole_direct.pth

# With video recording
python scripts/rl_games/train.py \
    --task Isaac-Cartpole-Direct-v0 \
    --video \
    --video_interval 2000
```

#### RSL-RL

```bash
# Basic training
python scripts/rsl_rl/train.py \
    --task Isaac-Cartpole-Direct-v0 \
    --num_envs 4096

# Resume training
python scripts/rsl_rl/train.py \
    --task Isaac-Cartpole-Direct-v0 \
    --resume \
    --load_run {timestamp}
```

### Evaluation Commands

```bash
# Test trained model (RL-Games)
python scripts/rl_games/play.py \
    --task Isaac-Cartpole-Direct-v0 \
    --checkpoint logs/rl_games/cartpole_direct/{timestamp}/nn/cartpole_direct.pth \
    --num_envs 1

# Record evaluation video
python scripts/rl_games/play.py \
    --task Isaac-Cartpole-Direct-v0 \
    --checkpoint logs/rl_games/cartpole_direct/{timestamp}/nn/cartpole_direct.pth \
    --video \
    --video_length 500
```

**Note**: Replace `{timestamp}` and `{experiment_name}` with your actual values. See [Finding Your Checkpoint Paths](#finding-your-checkpoint-paths) above.

---

## Dexhand Single Training

### Task Overview

**Environment**: `Template-Dexhand-Single-Direct-v0`

**Objective**: Control a dexterous hand with 6 DOF (L1, L2, L3, R1, R2, R3 joints) to maintain stable joint positions near rest state.

**Specifications**:
- **Actions**: Cable lengths (0.012m to 0.027m) for each joint
- **Observations**: Current joint positions (6 values)
- **Episode length**: 5.0 seconds (~600 steps at 120 Hz)

### Training Commands

```bash
# RL-Games
python scripts/rl_games/train.py \
    --task Template-Dexhand-Single-Direct-v0 \
    --num_envs 4096

# RSL-RL
python scripts/rsl_rl/train.py \
    --task Template-Dexhand-Single-Direct-v0 \
    --num_envs 4096
```

### Success Definition

The reward function defines success:

**Reward Components**:
1. **Alive Reward** (`+1.0` per step): Episode not terminated
2. **Termination Penalty** (`-2.0`): Early termination
3. **Joint Position Penalties** (`-1.0 × joint_angle²`): Deviation from zero for each of 6 joints

**Total Reward Formula**:
```
Reward = (1.0 × alive) + (-2.0 × terminated) + 
         Σ(-1.0 × joint_angle²) for all 6 joints
```

**Success Criteria**:
- ✅ Keep all 6 joints near zero (rest position)
- ✅ Avoid early termination (stay within ±π/2 for L2-L3, R1-R3; ±3.0 for L1)
- ✅ Complete full 5-second episodes

**Reward Interpretation**:
- **Maximum**: ~600 (all joints at 0, full episode)
- **High (>250)**: ✅ Excellent - Stable, joints near rest
- **Medium (150-250)**: ⚠️ Acceptable - Some deviation
- **Low (<150)**: ❌ Poor - Significant deviation or early termination

**Example**: Reward of **~268.8** indicates good performance - hand maintains stability with joints near rest positions.

### Evaluation

```bash
# Test trained model
python scripts/rl_games/play.py \
    --task Template-Dexhand-Single-Direct-v0 \
    --checkpoint logs/rl_games/dexhand_single/{timestamp}/nn/dexhand_single.pth \
    --num_envs 1

# Record video
python scripts/rl_games/play.py \
    --task Template-Dexhand-Single-Direct-v0 \
    --checkpoint logs/rl_games/dexhand_single/{timestamp}/nn/dexhand_single.pth \
    --video \
    --video_length 1000
```

### Configuration Files

- **RL-Games**: `source/Dexhand_single/.../agents/rl_games_ppo_cfg.yaml`
- **RSL-RL**: `source/Dexhand_single/.../agents/rsl_rl_ppo_cfg.py`
- **Environment**: `source/Dexhand_single/.../dexhand_single_env_cfg.py`

### Dexhand Training Videos

#### Training Progress

<!-- Early training -->
![Early Training](logs/rl_games/dexhand_single/{timestamp}/videos/train/rl-video-episode-0.mp4)

<!-- Mid training -->
![Mid Training](logs/rl_games/dexhand_single/{timestamp}/videos/train/rl-video-episode-2000.mp4)

#### Final Evaluation

<!-- Evaluation video -->
![Dexhand Evaluation](logs/rl_games/dexhand_single/{timestamp}/videos/play/rl-video-episode-0.mp4)

**To add videos**: Replace `{timestamp}` with your actual run timestamp (e.g., `2025-11-08_23-38-40`), or use GitHub video URLs:

```markdown
https://github.com/user-attachments/assets/{video_id}
```

---

## Recording Videos

### During Training

```bash
# Record videos periodically during training
python scripts/rl_games/train.py \
    --task Isaac-Cartpole-Direct-v0 \
    --num_envs 4096 \
    --video \
    --video_length 200 \
    --video_interval 2000
```

**Parameters:**
- `--video`: Enable video recording
- `--video_length`: Number of steps to record per video (default: 200)
- `--video_interval`: Record a video every N steps (default: 2000)

**Video Location:**
```
logs/rl_games/{experiment_name}/{timestamp}/videos/train/
└── rl-video-episode-{step}.mp4
```

#### RSL-RL

```bash
# Record videos during training
python scripts/rsl_rl/train.py \
    --task Isaac-Cartpole-Direct-v0 \
    --num_envs 4096 \
    --video \
    --video_length 200 \
    --video_interval 2000
```

**Parameters**:
- `--video`: Enable recording
- `--video_length`: Steps per video (default: 200)
- `--video_interval`: Record every N steps (default: 2000)

### During Evaluation

```bash
# Record evaluation video
python scripts/rl_games/play.py \
    --task Isaac-Cartpole-Direct-v0 \
    --checkpoint logs/rl_games/cartpole_direct/{timestamp}/nn/cartpole_direct.pth \
    --video \
    --video_length 500
```

### Video Locations

- **Training**: `logs/{framework}/{experiment}/{timestamp}/videos/train/`
- **Evaluation**: `logs/{framework}/{experiment}/{timestamp}/videos/play/`

### Embedding Videos in README

**GitHub Video Format** (recommended):

```markdown
https://github.com/user-attachments/assets/{video_id}
```

**Example Videos**:

Cartpole training to result demonstration:

https://github.com/user-attachments/assets/1fae4344-a282-49c4-89c6-de5099e02a33

[Screencast from 11-08-2025 03:37:29 PM.webm](https://github.com/user-attachments/assets/8a63037d-7bf9-4ffc-af26-bfe2f41a2ea8)


Dexhand training and play demonstration:

[Screencast from 11-08-2025 11:48:39 PM.webm](https://github.com/user-attachments/assets/b0403223-865e-47b1-8f12-f8a26942cf2e)


Cartpole evaluation:

```bash
# 1. Train with periodic video recording
python scripts/rl_games/train.py \
    --task Isaac-Cartpole-Direct-v0 \
    --num_envs 4096 \
    --video \
    --video_interval 5000

# 2. After training, record a high-quality evaluation video
python scripts/rl_games/play.py \
    --task Isaac-Cartpole-Direct-v0 \
    --checkpoint logs/rl_games/cartpole_direct/{timestamp}/nn/cartpole_direct.pth \
    --video \
    --video_length 1000 \
    --num_envs 1
```

---

## Additional Resources

**Note**: When `--video` is enabled, cameras are automatically enabled in the environment. Make sure your task configuration includes camera sensors if you want to record visual observations.

---

## Summary

Both **RL-Games** and **RSL-RL** are powerful frameworks for training RL agents in Isaac Sim:

- **RL-Games**: YAML configs, registry-based, flexible for multiple algorithms
- **RSL-RL**: Python configs, direct API, type-safe, simpler code

Choose based on your preference for configuration style and API complexity. Both work seamlessly with Isaac Lab environments!
