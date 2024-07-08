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

**Note**: You may have to move the `kan.py` file from the `efficient-kan` repository to the `src` directory of this project, using this command. 
```bash
mv efficient-kan/kan.py KAN-It-Play-Flappy-Bird/src/kan.py
```

## Usage
To run the project and train the model, simply run the following command:
```bash
python main.py --env env_name --train
```
You can remove the `--train` flag to test the model without training it. `env_name` is the name of the environment you want to use (hint: it should be the first header of `config.yaml` file).

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

## Experiment results
The following table shows the comparison between the MLP model and the KAN model in terms of training efficiency and speed.
### Summary Table

| Metric                       | MLP Model                     | KAN Model                     |
|------------------------------|-------------------------------|-------------------------------|
| **Number of Hidden Units**   | 512                           | 32                            |
| **Mean Reward at Latest Episode** | **2.71**                | 1.89                          |
| **Maximum Episodes Taken**   | 103,504                       | **36,807**                    |
| **Average Time per Episode** | 0.22s                         | **0.12s**                     |

### Analysis

The MLP model, with 512 hidden units, exhibits higher mean rewards at the latest episode. However, the KAN model, with only 32 hidden units, can achieve a comparable performance with fewer hidden units. However, the KAN model takes almost three times the number of episodes as the MLP model to reach the same performance level, which perfectly explains why KAN is considered "inefficient to train" compared to MLP by many early tester. On the flip side, this is an early implementation of KAN model, and it's still a long way to go to optimize it! 

## So, KAN it play Flappy Bird?
Yes, it KAN! The KAN model demonstrates its capability to play Flappy Bird effectively, achieving good performance with fewer hidden units and faster convergence compared to the MLP model. However, further experiments and optimizations can be conducted to enhance the model's performance and explore its full potential in Reinforcement Learning. And if you pit KAN and MLP against each other, the **MLP** model will win easily!

## Credits
This project is inspired by the [Flappy Bird DQN](https://www.youtube.com/watch?v=Ejv8yv5-i0M) tutorial by [Tech with Tim](https://www.youtube.com/channel/UC4JX40jDee_tINbkjycV4Sg). The KAN model implementation is based on the [EfficientKAN](https://github.com/Blealtan/efficient-kan/tree/master).





