# KAN It Play Flappy Bird?
Credits: Dinh Ngoc An

## Introduction
This is a simple project to test the capabilities of the [KAN](https://arxiv.org/abs/2404.19756) (Kolmogorov-Arnold Networks) model on a simple Flappy Bird game using Reinforcement Learning. Here, the RL algorithm used is Deep Q-Network (DQN) with the original Linear layer replaced with KAN model.

## Installation
1. Clone the repository
Clone the `EfficientKAN` repository, which is the latest and most efficient implementation of KAN model using PyTorch (by July 2024).
```bash
git clone  https://github.com/Blealtan/efficient-kan.git
```

2. Install the required packages

We recommend using a virtual environment to install the required packages and avoid dependencies issue (trust me, it's frustrating). You can create a virtual environment using the following command:
```bash
python -m venv venv
```
Then, activate the virtual environment:
```bash
source venv/bin/activate
```
Now, install the required packages using the following command:
```bash
pip install -r requirements.txt
```

## Usage
To run the project, simply run the following command:
```bash
python main.py --train
```

## Hyperparameters tuning
You can tune the hyperparameters in the `config.yaml` file. Here are some important hyperparameters that you might want to tune:
```yaml
    env_id: Environment ID (flappy_bird)
    experience_replay_size: Size of the experience replay buffer
    batch_size: Batch size
    epsilon_start: Initial epsilon value
    epsilon_end: Final epsilon value
    epsilon_decay: Epsilon decay rate after each episode
    network_update_frequency: Frequency of updating the target network
    learning_rate: Learning rate
    discount_factor: Discount factor (default: 0.99)
    stop_on_reward: Stop training when the average reward reaches this value
    model_params:
        model_type: Either 'kan' or 'mlp' for experiment
        num_hidden_units: Number of hidden units in either of those model types (note that we only have one hidden layer).
    env_params:
        use_lidar: Whether to use lidar as input to the model (default: False)
```



