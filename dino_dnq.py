import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import cv2
import pyautogui
from collections import deque
from PIL import Image
import time
import matplotlib.pyplot as plt
import logging
import sys
import os
import json


# LOGGING STUFF 
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# A File handler for debug logs
file_handler = logging.FileHandler("debug.log", mode="w")
file_handler.setLevel(logging.DEBUG)

# A Console handler for info-level logs
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)

# Format for both
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - [%(levelname)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)


# DEFINING  HYPERPARAMETERS
GAMMA = 0.99
LEARNING_RATE = 1e-4
BATCH_SIZE = 32
REPLAY_MEMORY_SIZE = 50000
START_TRAINING_AFTER = 1000
TARGET_UPDATE_INTERVAL = 1000

# These will be loaded/resumed if found in train_state.json
epsilon = 1.0
epsilon_min = 0.1
epsilon_decay = 0.995

# Consecutive-frame game-over detection
STUCK_THRESHOLD = 0.82 #STILL ADJUSTING THESE 2
STUCK_LIMIT = 2

# Checkpoint paths
MODEL_PATH = "dino_dqn_model_latest.pth"
STATE_PATH = "train_state.json"


# DQN MODEL   
class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(self._get_conv_out(input_shape), 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        x = x.float() / 255  # Normalize input
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)





def grab_screen(region=None):
    screen = pyautogui.screenshot(region=region)
    frame = np.array(screen)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.resize(frame, (84, 84))
    return frame

def jump():
    pyautogui.keyDown("space")
    time.sleep(0.1)
    pyautogui.keyUp("space")


class ReplayMemory:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.stack(state), action, reward, np.stack(next_state), done
    
    def __len__(self):
        return len(self.buffer)


#   TRAINING FUNCTION
def train_dqn(env_region):
    global epsilon  # so we can resume from loaded state
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu") # checking for GPU on MAC
    input_shape = (1, 84, 84)
    num_actions = 2

    # Build networks
    policy_net = DQN(input_shape, num_actions).to(device)
    target_net = DQN(input_shape, num_actions).to(device)

    # Load previous model checkpoint if exists
    if os.path.exists(MODEL_PATH):
        policy_net.load_state_dict(torch.load(MODEL_PATH))
        logger.info(f"Resumed model from {MODEL_PATH}")
    else:
        logger.info("No previous model found. Starting fresh.")

    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    # Optimizer
    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    memory = ReplayMemory(REPLAY_MEMORY_SIZE)

    # We'll also resume state (including epsilon, steps_done) if it exists
    steps_done = 0
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH, "r") as f:
            state_data = json.load(f)
        epsilon = state_data.get("epsilon", epsilon)
        steps_done = state_data.get("steps_done", 0)
        logger.info(f"Resumed state: epsilon={epsilon}, steps_done={steps_done}")
    else:
        logger.info("No train_state.json found. Starting with default hyperparameters.")

    rewards = []
    game_counter = 0

    for episode in range(1000):
        logger.info(f"Episode {episode} started...")

        total_reward = 0
        state_frame = grab_screen(env_region)
        state_frame = np.expand_dims(state_frame, axis=0)
        state = state_frame

        prev_frame = None
        stuck_frames = 0

        for t in range(10000):
            # Epsilon-greedy action selection
            state_tensor = torch.tensor(
                np.expand_dims(state, axis=0),
                device=device, dtype=torch.float32
            )
            with torch.no_grad():
                q_values = policy_net(state_tensor)
            
            if random.random() > epsilon:
                action = q_values.max(1)[1].item()
            else:
                action = random.randrange(num_actions)
            
            if action == 1:
                jump()

            next_frame = grab_screen(env_region)
            next_frame = np.expand_dims(next_frame, axis=0)

            done = False
            reward = 1.0  # living reward

            if prev_frame is not None:
                diff = np.mean(np.abs(next_frame - prev_frame))
                logger.debug(f"Ep={episode}, Step={t}, diff={diff:.2f}, stuck={stuck_frames}")

                if diff < STUCK_THRESHOLD:
                    stuck_frames += 1
                else:
                    stuck_frames = 0

                if stuck_frames > STUCK_LIMIT:
                    done = True
                    reward = -100.0

            prev_frame = next_frame

            memory.push(state, action, reward, next_frame, done)
            state = next_frame
            total_reward += reward

            # Train if memory is large enough
            if len(memory) > START_TRAINING_AFTER:
                batch = memory.sample(BATCH_SIZE)
                train_batch(batch, policy_net, target_net, optimizer, device)

            # Update target net periodically
            if steps_done % TARGET_UPDATE_INTERVAL == 0:
                target_net.load_state_dict(policy_net.state_dict())

            steps_done += 1

            if done:
                game_counter += 1
                logger.info(f"Game {game_counter} finished at step {t}.")
                break

        # Epsilon decay
        rewards.append(total_reward)
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        logger.info(f"Episode {episode}, Reward={total_reward}, Epsilon={epsilon}, Games={game_counter}")

    
        # SAVEING THE MODEL AFTER EVERY 40 GAMES
        if (episode + 1) % 40 == 0:
            torch.save(policy_net.state_dict(), MODEL_PATH)
            logger.info(f"Checkpoint saved at episode {episode + 1}")

            # Save the training state (epsilon, steps_done)
            with open(STATE_PATH, "w") as f:
                json.dump({"epsilon": epsilon, "steps_done": steps_done}, f)
        else:
            # Optionally still update train_state after each ep, if you prefer
            with open(STATE_PATH, "w") as f:
                json.dump({"epsilon": epsilon, "steps_done": steps_done}, f)

    # Plot final rewards
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title(f"DQN Training Performance - {game_counter} Games Played")
    plt.show()

#   TRAIN BATCH UPDATE 
def train_batch(batch, policy_net, target_net, optimizer, device):
    state, action, reward, next_state, done = batch

    state = torch.tensor(state, device=device, dtype=torch.float32)
    next_state = torch.tensor(next_state, device=device, dtype=torch.float32)
    action = torch.tensor(action, device=device, dtype=torch.int64)
    reward = torch.tensor(reward, device=device, dtype=torch.float32)
    done = torch.tensor(done, device=device, dtype=torch.float32)

    q_values = policy_net(state).gather(1, action.unsqueeze(-1)).squeeze(-1)
    next_q_values = target_net(next_state).max(1)[0]
    expected_q_values = reward + (1 - done) * GAMMA * next_q_values

    loss = nn.MSELoss()(q_values, expected_q_values.detach())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


if __name__ == "__main__":
    # UNCOMMENT THE BELOW CODE TO GET EXACT COORDINATES(env_region) OF THE GAME WINDOW
    # Hard-coded region. Modify if needed or add your own logic to pick corners.
    # Hard-code region or un-comment these lines to select region:
    # print("Move your mouse to the top-left and bottom-right corners of the game window to set the capture region.")
    # time.sleep(10)
    # top_left = pyautogui.position()
    # print(f"Top-left corner: {top_left}")
    # time.sleep(10)
    # bottom_right = pyautogui.position()
    # print(f"Bottom-right corner: {bottom_right}")
    # env_region = (
    #     top_left.x,
    #     top_left.y,
    #     bottom_right.x - top_left.x,
    #     bottom_right.y - top_left.y
    # )
    # print(f"Screen capture region: {env_region}")
    
    env_region = (1181, 165, 533, 150)
    train_dqn(env_region)


