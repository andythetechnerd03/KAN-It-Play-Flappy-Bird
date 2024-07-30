import flappy_bird_gymnasium
import gymnasium
import torch
import itertools
import yaml
import numpy as np
import os
import matplotlib
from matplotlib import pyplot as plt
from datetime import datetime, timedelta
import argparse
from typing import List
from tqdm import tqdm
from pygame_screen_record import ScreenRecorder

from src.dqn import DeepQNetwork
from src.experience_replay import ExperienceReplay

# For printing date and time
DATE_FORMAT = "%m-%d_%H-%M-%S"

# Directory for saving run info
RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)

matplotlib.use("Agg")

SEED=42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
# Create the environment

class Agent:
    def __init__(self, env: str, device: str="cpu") -> None:
        """ Initialize the Agent class
        Args:
            env (str): environment to run the agent
            device (str): device to train the agent on, defaults to "cpu"
        """
        self.device = device
        assert device in ["cpu","cuda"], "device must be either cpu or cuda"

        self.output_dir = os.path.join(RUNS_DIR, env)
        os.makedirs(self.output_dir, exist_ok=True)
        self.env = env

        with open("config.yml", "r") as file:
            config = yaml.safe_load(file)
            self.config = config[env]
        
        self.experience_replay_size = self.config["experience_replay_size"]
        self.env_id = self.config["env_id"]
        self.batch_size = self.config["batch_size"]
        self.epsilon_start = self.config["epsilon_start"]
        self.epsilon_end = self.config["epsilon_end"]
        self.epsilon_decay = self.config["epsilon_decay"]
        self.network_update_frequency = self.config["network_update_frequency"]
        self.learning_rate = self.config["learning_rate"]
        self.discount_factor = self.config["discount_factor"]
        self.stop_on_reward = self.config["stop_on_reward"]
        self.num_hidden_units = self.config["model_params"]["num_hidden_units"]
        self.model_type = self.config["model_params"]["model_type"]
        self.env_params = self.config.get("env_params",{})
        self.max_episodes = self.config["max_episodes"]

        # Training info
        self.loss_fn = torch.nn.MSELoss()

        # Path to run info
        self.LOG_FILE = os.path.join(self.output_dir, f"{env}.log")
        self.MODEL_FILE = os.path.join(self.output_dir, f"{env}.pt")
        self.GRAPH_FILE = os.path.join(self.output_dir, f"{env}.png")


    def run(self, train: bool = False, render: bool = False):
        """ Run the agent
        Args:
            train (bool): train the agent if True
            render (bool): render the environment if True
        """
        # initialize training
        if train:
            start_time = datetime.now()
            last_graph_update_time = start_time

            log_message = f"{start_time.strftime(DATE_FORMAT)}: Training started \nConfiguration: {self.config}"
            self.log(log_message, save_to_file=True)
        
        # Create an instance of the environment
        env = gymnasium.make(self.env_id, render_mode="human" if render else None, **self.env_params)

        num_states = env.observation_space.shape[0] # Number of states
        num_actions = env.action_space.n # Number of actions

        rewards_per_episode = [] # Keep track of rewards per episode

        # Declare policy DQN
        policy_dqn = DeepQNetwork(num_states, num_actions,
                                  self.num_hidden_units, self.model_type).to(self.device)
        # If training mode, then declare replay buffer
        if train:
            # Declare experience replay buffer
            replay_buffer = ExperienceReplay(max_size=self.experience_replay_size)

            # Initialize epsilon value
            epsilon = self.epsilon_start

            # Declare target DQN and load parameters from policy DQN
            target_dqn = DeepQNetwork(num_states, num_actions,
                                      self.num_hidden_units, self.model_type).to(self.device)
            target_dqn.load_state_dict(policy_dqn.state_dict())

            # Track steps
            steps = 0

            # Declare optimizer
            self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate)

            # Track best reward
            best_reward = -np.inf
        
        else:
            # Load model from directory
            print(f"Loading model from {self.MODEL_FILE}")
            policy_dqn.load_state_dict(torch.load(self.MODEL_FILE))

            # Put model to eval mode
            policy_dqn.eval()

        # Run for max_episodes episodes, break when stop_on_reward is reached
        pbar = tqdm(range(self.max_episodes), desc="Episodes", unit="ep")
        for episode in pbar:
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float32).to(self.device)
            terminated = False
            episode_reward = 0

            while (not terminated and episode_reward < self.stop_on_reward):
                # Next action
                action = self.choose_action_(env, state, policy_dqn, train, epsilon)

                # Environment step
                next_state, reward, terminated, _, info = env.step(action.item())

                # Update episode reward
                episode_reward += reward

                # Convert to tensors
                next_state = torch.tensor(next_state, dtype=torch.float32).to(self.device)
                reward = torch.tensor(reward, dtype=torch.float32).to(self.device)

                # If training mode, then add to replay buffer
                if train:
                    replay_buffer.add(state, action, reward, next_state, terminated) 

                    steps += 1

                state = next_state

            rewards_per_episode.append(episode_reward)
            pbar.set_postfix({"Reward": episode_reward})

            # Save model if reward is greater than best reward
            if train:
                if episode_reward > best_reward:
                    log_message = f"{datetime.now().strftime(DATE_FORMAT)}: Episode {episode}, New best reward: {episode_reward}"
                    self.log(log_message, save_to_file=True)

                    torch.save(policy_dqn.state_dict(), self.MODEL_FILE)
                    best_reward = episode_reward

                # Update graph every 1000 episodes
                if episode % 1000 == 0:
                    self.save_graph(rewards_per_episode)

                # If replay buffer has enough samples, then sample and train
                if len(replay_buffer) > self.batch_size:
                    batch = replay_buffer.sample(self.batch_size)
                    self.optimize(batch, policy_dqn, target_dqn)
                    
                    # Perform epsilon value decay
                    epsilon = max(self.epsilon_end, epsilon * self.epsilon_decay)

                    # Copy policy DQN parameters to target DQN after some steps and reset steps
                    if steps > self.network_update_frequency:
                        target_dqn.load_state_dict(policy_dqn.state_dict())
                        steps = 0

    def optimize(self, batch, policy_dqn, target_dqn):
        """ Optimize the policy DQN
        Args:
            batch (list): batch of experiences
            policy_dqn (DeepQNetwork): policy DQN
            target_dqn (DeepQNetwork): target DQN
        """
        states, actions, rewards, next_states, terminations = zip(*batch)

        # Convert to batch tensors for faster computation
        states = torch.stack(states).to(self.device)
        actions = torch.stack(actions).to(self.device)
        rewards = torch.stack(rewards).to(self.device)
        next_states = torch.stack(next_states).to(self.device)
        terminations = torch.tensor(terminations, dtype=torch.float32).to(self.device)

        # Get Q values
        q_values = policy_dqn(states)
        q_values = q_values.gather(1, actions.unsqueeze(dim=1)).squeeze()

        # Get target Q values
        with torch.no_grad():
            target_q_values = target_dqn(next_states)
            target_q_values = rewards + self.discount_factor * target_q_values.max(dim=1).values * (1 - terminations)

        # Compute loss
        loss = self.loss_fn(q_values, target_q_values)

        # Zero gradients
        self.optimizer.zero_grad()

        # Backward pass
        loss.backward()

        # Update weights
        self.optimizer.step()

    def save_graph(self, rewards_per_episode: List[float]):
        # Save graph
        fig = plt.figure(1)

        # Plot average rewards per episode
        mean_rewards = [np.mean(rewards_per_episode[max(0, i-99):i+1]) for i in range(len(rewards_per_episode))]

        plt.plot(rewards_per_episode, label="Reward per episode")
        plt.plot(mean_rewards, label="Mean reward (100 episodes)")
        plt.xlabel("Episodes")
        plt.ylabel("Reward")
        plt.title(f"Reward per episode - Model {self.model_type}")
        plt.legend()

        # Save figure
        plt.savefig(self.GRAPH_FILE)
        plt.close(fig)

    def log(self, message: str, save_to_file: bool) -> None:
        """ Log message to console and optionally to file
        Args:
            message (str): message to log
            save_to_file (bool): save to file if True

        Returns: None
        """
        print(message)
        if save_to_file:
            with open(self.LOG_FILE, "a") as file:
                file.write(message + "\n")

    def test(self, render: bool=True, num_episodes: int=10, print_score: bool=False) -> List[int]:
        """ Test the agent
        Args:
            render (bool): display Pygame window if True
            num_episodes (int): number of episodes to test
            print_score (bool): whether to print score after each episode, defaults to False
        
        Returns: List[int]: rewards per episode
        """
        # Load model from directory
        env = gymnasium.make(self.env_id, render_mode="human" if render else None, **self.env_params)
        print("Initialized environment")
        num_states = env.observation_space.shape[0] # Number of states
        num_actions = env.action_space.n # Number of actions

        # Declare policy DQN
        policy_dqn = DeepQNetwork(num_states, num_actions,
                                  self.num_hidden_units, self.model_type).to(self.device)
        print(f"Loading model from {self.MODEL_FILE}")
        policy_dqn.load_state_dict(torch.load(self.MODEL_FILE))
        # Put model to eval mode
        policy_dqn.eval()
        rewards_per_episode = []
        for episode in tqdm(range(num_episodes), desc="Episodes", unit="ep"):
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float32).to(self.device)
            terminated = False
            score = 0

            # Initialize the recorder
            if render:
                recorder = ScreenRecorder(fps=30)
                recorder.start_rec()

            while not terminated:
                action = self.choose_action_(env, state, policy_dqn, train=False, epsilon=0.0)

                next_state, reward, terminated, _, info = env.step(action.item())

                state = torch.tensor(next_state, dtype=torch.float32).to(self.device)

                if reward == 1: score += 1

            # Stop recording and send output to streamlit
            if render:
                recorder.stop_rec()
                recorder.save_recording(f"videos/{self.env}.mp4")

            if print_score: print("Score:", score)
            rewards_per_episode.append(score)

        if print_score: print(f"High score: {max(rewards_per_episode)}")
        return rewards_per_episode
    
    def choose_action_(self, env: gymnasium.Env,
                       state: torch.Tensor,
                       policy_dqn: DeepQNetwork,
                       train: bool, epsilon: float) -> torch.Tensor:
        """ Choose action using epsilon-greedy algorithm
        Args:
            env (gym.Env): environment
            state (torch.Tensor): state
            policy_dqn (DeepQNetwork): policy DQN
            train (bool): training mode
            epsilon (float): epsilon value
        
        Returns:
            torch.Tensor: action
        """
        if train and np.random.rand() < epsilon: 
            action = env.action_space.sample()
            action = torch.tensor(action, dtype=torch.int64).to(self.device)
        else:
            # Get action from policy, add batch dimension 
            with torch.no_grad(): action = policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax()
        return action
        

def parse_args():
    parser = argparse.ArgumentParser(description="Train or test DQN agent")
    parser.add_argument("--env", type=str, default="flappy_bird", help="Environment to run the agent")
    parser.add_argument("--train", action="store_true", help="Train the agent")
    return parser.parse_args()


# Test
if __name__ == "__main__":
    args = parse_args()
    agent = Agent(args.env)
    if args.train:
        agent.run(train=args.train, render=False)
    else:
        agent.run(train=args.train, render=True)