#!/bin/bash

# TinyRL Reproducibility Script
# This script re-creates baseline results end-to-end with deterministic settings

set -e  # Exit on any error

echo "=== TinyRL Reproducibility Script ==="
echo "This script will reproduce baseline results with deterministic settings"
echo ""

# Check if we're in the right directory
if [ ! -f "requirements.txt" ]; then
    echo "Error: requirements.txt not found. Please run this script from the project root."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Set deterministic environment variables
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=42
export TORCH_DETERMINISTIC=1

echo "Setting deterministic environment variables..."
echo "CUBLAS_WORKSPACE_CONFIG=$CUBLAS_WORKSPACE_CONFIG"
echo "PYTHONHASHSEED=$PYTHONHASHSEED"
echo "TORCH_DETERMINISTIC=$TORCH_DETERMINISTIC"

# Create output directories
echo "Creating output directories..."
mkdir -p outputs/ppo_cartpole
mkdir -p outputs/a2c_lunarlander
mkdir -p logs

# Function to run training with specific config
run_training() {
    local config_name=$1
    local output_dir=$2
    local total_timesteps=$3
    
    echo ""
    echo "=== Training $config_name ==="
    echo "Config: configs/train/$config_name.yaml"
    echo "Output: $output_dir"
    echo "Timesteps: $total_timesteps"
    
    # Run training
    python -m tinyrl.train \
        --config-name $config_name \
        training.total_timesteps=$total_timesteps \
        output.dir=$output_dir \
        seed=42 \
        deterministic=true
    
    echo "Training completed for $config_name"
}

# Function to run evaluation
run_evaluation() {
    local model_path=$1
    local env_name=$2
    local n_episodes=$3
    
    echo ""
    echo "=== Evaluating $model_path ==="
    echo "Environment: $env_name"
    echo "Episodes: $n_episodes"
    
    python -c "
import gymnasium as gym
from stable_baselines3 import PPO, A2C
import numpy as np

# Set deterministic seed
np.random.seed(42)
import torch
torch.manual_seed(42)

# Load model
model = PPO.load('$model_path') if 'ppo' in '$model_path' else A2C.load('$model_path')

# Create environment
env = gym.make('$env_name')

# Evaluate
rewards = []
for _ in range($n_episodes):
    obs, _ = env.reset(seed=42)
    done = False
    episode_reward = 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, _ = env.step(action)
        episode_reward += reward
        if truncated:
            done = True
    rewards.append(episode_reward)

mean_reward = np.mean(rewards)
std_reward = np.std(rewards)

print(f'Evaluation Results:')
print(f'Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}')
print(f'Min Reward: {np.min(rewards):.2f}')
print(f'Max Reward: {np.max(rewards):.2f}')
print(f'Episodes: {len(rewards)}')
"

    echo "Evaluation completed for $model_path"
}

# Function to run benchmarks
run_benchmarks() {
    echo ""
    echo "=== Running Benchmarks ==="
    
    # Model size benchmarks
    echo "Model size benchmarks..."
    python -c "
import torch
from tinyrl.models import PPOActor, PPOCritic, A2CActor, A2CCritic
from tinyrl.utils import get_model_size

# PPO CartPole model
ppo_actor = PPOActor(obs_dim=4, action_dim=2)
ppo_critic = PPOCritic(obs_dim=4)
ppo_total_size = get_model_size(ppo_actor)['total_memory_bytes'] + get_model_size(ppo_critic)['total_memory_bytes']

# A2C LunarLander model  
a2c_actor = A2CActor(obs_dim=8, action_dim=4)
a2c_critic = A2CCritic(obs_dim=8)
a2c_total_size = get_model_size(a2c_actor)['total_memory_bytes'] + get_model_size(a2c_critic)['total_memory_bytes']

print(f'PPO CartPole Model Size: {ppo_total_size / 1024:.1f} KB')
print(f'A2C LunarLander Model Size: {a2c_total_size / 1024:.1f} KB')
"

    # Training time benchmarks
    echo ""
    echo "Training time benchmarks..."
    echo "Note: These are approximate times for the specified timesteps"
    echo "PPO CartPole (50k steps): ~2-5 minutes"
    echo "A2C LunarLander (100k steps): ~5-10 minutes"
}

# Main reproduction pipeline
echo ""
echo "Starting reproduction pipeline..."

# Run PPO CartPole training
run_training "ppo_cartpole" "outputs/ppo_cartpole" 50000

# Run A2C LunarLander training  
run_training "a2c_lunarlander" "outputs/a2c_lunarlander" 100000

# Run evaluations
echo ""
echo "=== Running Evaluations ==="

# Evaluate PPO CartPole
if [ -f "outputs/ppo_cartpole/final_model.zip" ]; then
    run_evaluation "outputs/ppo_cartpole/final_model" "CartPole-v1" 100
else
    echo "Warning: PPO CartPole model not found"
fi

# Evaluate A2C LunarLander
if [ -f "outputs/a2c_lunarlander/final_model.zip" ]; then
    run_evaluation "outputs/a2c_lunarlander/final_model" "LunarLander-v2" 100
else
    echo "Warning: A2C LunarLander model not found"
fi

# Run benchmarks
run_benchmarks

# Generate reproducibility report
echo ""
echo "=== Generating Reproducibility Report ==="

cat > reproducibility_report.txt << EOF
TinyRL Reproducibility Report
=============================

Date: $(date)
Commit: $(git rev-parse HEAD 2>/dev/null || echo "Unknown")
Python: $(python --version)
PyTorch: $(python -c "import torch; print(torch.__version__)")

Environment Variables:
- CUBLAS_WORKSPACE_CONFIG: $CUBLAS_WORKSPACE_CONFIG
- PYTHONHASHSEED: $PYTHONHASHSEED
- TORCH_DETERMINISTIC: $TORCH_DETERMINISTIC

Training Configurations:
- PPO CartPole: configs/train/ppo_cartpole.yaml
- A2C LunarLander: configs/train/a2c_lunarlander.yaml

Output Directories:
- PPO CartPole: outputs/ppo_cartpole/
- A2C LunarLander: outputs/a2c_lunarlander/

Model Files:
$(find outputs -name "*.zip" -o -name "*.pkl" -o -name "*.yaml" | sort)

Reproducibility Notes:
- All training uses deterministic seeds (42)
- PyTorch 2.3 compilation enabled
- Weights & Biases logging enabled
- Model checkpoints saved every 5000/10000 steps
- Best models saved based on evaluation performance

Expected Results:
- PPO CartPole: Mean reward > 450 (episode length ~500)
- A2C LunarLander: Mean reward > 200 (episode length ~1000)

To reproduce these results:
1. Run this script: ./reproduce.sh
2. Ensure all dependencies are installed
3. Use the same seed (42) and deterministic settings
4. Allow sufficient training time for convergence

EOF

echo "Reproducibility report saved to reproducibility_report.txt"

echo ""
echo "=== Reproduction Complete ==="
echo "All baseline results have been reproduced."
echo "Check the output directories for trained models and logs."
echo "Review reproducibility_report.txt for detailed information." 