import copy
import itertools
import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed

import gymnasium as gym
import pandas as pd
import torch
import torch.optim as optim

from model import QAgent
from replay_buffer import ReplayBuffer


torch.set_num_threads(1)
os.environ["OMP_NUM_THREADS"] = "1"


# Hyperparameters fixed - these are the best values found in 2.2
LEARNING_RATE = 0.001
GAMMA = 0.99
EPSILON = 0.1
NETWORK_SIZE = 128
UPDATE_FREQUENCY = 4
N_STEPS = 10 ** 6
SEEDS = [67, 69, 420, 666, 69420]

# Replay buffer settings
BUFFER_CAPACITY = 50_000   
BATCH_SIZE = 64       
WARMUP_STEPS = 1_000


def evaluate(agent, env, episodes=30):
    # Run the agent greedily for a fixed number of episodes and return the mean undiscounted return
    total_rewards = []

    for _ in range(episodes):
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32)
        ep_reward = 0.0

        for _ in range(500):
            with torch.no_grad():
                action = torch.argmax(agent(state)).item()

            next_state, reward, done, truncated, _ = env.step(action)
            ep_reward += reward

            if done or truncated:
                break

            state = torch.tensor(next_state, dtype=torch.float32)

        total_rewards.append(ep_reward)

    return sum(total_rewards) / len(total_rewards)


def train(env, eval_env, agent, optimizer, criterion,
          use_target_network, use_experience_replay):
    # loop for the four DQN configurations
    # Set up optional components depending on the configuration

    if use_target_network:
        # Copy the online network and freeze it from gradient updates.
        # We will periodically hard-sync it with the online network.
        target_net = copy.deepcopy(agent)
        target_net.eval()
    else:
        target_net = None

    if use_experience_replay:
        replay_buffer = ReplayBuffer(capacity=BUFFER_CAPACITY)
    else:
        replay_buffer = None
        

    state, _ = env.reset()
    state     = torch.tensor(state, dtype=torch.float32)

    optimizer.zero_grad()
    learning_curve = []

    for step in range(N_STEPS):

        # epsilon-greedy
        with torch.no_grad():
            q_values = agent(state)

        if random.random() < EPSILON:
            
           action = env.action_space.sample()
        else:
            action = torch.argmax(q_values).item()
            

       # Step the environment
        next_state, reward, done, truncated, _ = env.step(action)
        next_state = torch.tensor(next_state, dtype=torch.float32)

        # Update the agent
        if use_experience_replay:
            replay_buffer.store(state, action, reward, next_state, done)

            # Only start updating once the buffer has enough transitions
            if len(replay_buffer) >= WARMUP_STEPS:
                update_from_replay(agent, target_net, optimizer, criterion,
                                   replay_buffer, step)
        else:
            update_online(agent, target_net, optimizer, criterion,
                         state, action, reward, next_state, done, step)

        # Advance to the next state 
        state = next_state
        if done or truncated:
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float32)

        # Evaluate every 5000 steps 
        if (step + 1) % 5000 == 0:
            score = evaluate(agent, eval_env)
            learning_curve.append((step, score))

    return agent, learning_curve


def update_online(agent, target_net, optimizer, criterion,
                  state, action, reward, next_state, done, step):
    # Single-step online TD update (used by naive and tn only).
     
    # Use the target network for bootstrapping if available
    bootstrap_net = target_net if target_net is not None else agent

    predicted_q = agent(state)

    with torch.no_grad():
        q_next = bootstrap_net(next_state)

    if done:
        target = float(reward)
    else:
        target = reward + GAMMA * q_next.max().item()

    loss = criterion(predicted_q[action], torch.tensor(target, dtype=torch.float32))

    # Accumulate gradients over UPDATE_FREQUENCY steps before applying them
    (loss / UPDATE_FREQUENCY).backward()
    if (step + 1) % UPDATE_FREQUENCY == 0:
        optimizer.step()
        optimizer.zero_grad()


def update_from_replay(agent, target_net, optimizer, criterion, replay_buffer, step):
    # TD update sampled from the replay buffer (used by er and tn_er).
    
    states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH_SIZE)

    # Predicted Q-values for the actions that were actually taken
    predicted_q = agent(states).gather(1, actions.unsqueeze(1)).squeeze(1)

    # TD targets – use target net if available, otherwise online net
    bootstrap_net = target_net if target_net is not None else agent
    with torch.no_grad():
        best_next_q = bootstrap_net(next_states).max(dim=1).values
    targets = rewards + GAMMA * best_next_q * (1.0 - dones)

    loss = criterion(predicted_q, targets)
    optimizer.zero_grad()   # clean slate every update
    loss.backward()
    optimizer.step()

def run_experiment(config, seed):
    # Initialise a fresh agent and environment and run a full training run."""
    use_target_network = config in ("only_tn", "tn_er")
    use_experience_replay = config in ("only_er", "tn_er")

    torch.manual_seed(seed)
    random.seed(seed)

    env = gym.make("CartPole-v1")
    eval_env = gym.make("CartPole-v1")
    env.action_space.seed(seed)

    agent = QAgent(hidden_size=NETWORK_SIZE)
    optimizer = optim.Adam(agent.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.MSELoss()

    _, learning_curve = train(env, eval_env, agent, optimizer, criterion,
                              use_target_network, use_experience_replay)

    return {"config": config, "seed": seed, "curve": learning_curve}


if __name__ == "__main__":
    configs = ["naive", "only_tn", "only_er", "tn_er"]
    experiments = list(itertools.product(configs, SEEDS))

    # Skip experiments that are already saved to disk
    csv_file  = "results_dqn.csv"
    completed = set()
    if os.path.exists(csv_file):
        existing = pd.read_csv(csv_file)
        for _, row in existing[["Config", "Seed"]].drop_duplicates().iterrows():
            completed.add((row["Config"], int(row["Seed"])))

    experiments = [e for e in experiments if e not in completed]
    print(f"Running {len(experiments)} experiments ({len(completed)} already done)")

    with ProcessPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(run_experiment, *exp): exp for exp in experiments}

        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()

            rows = [
                {"Config": result["config"], "Seed": result["seed"],
                 "Step": step, "Score": score}
                for step, score in result["curve"]
            ]

            df = pd.DataFrame(rows)
            file_exists = os.path.exists(csv_file)
            df.to_csv(csv_file, mode="a", header=not file_exists, index=False)

            print(f"[{i}/{len(experiments)}] {result['config']:8s}  "
                  f"seed={result['seed']}  "
                  f"final_score={result['curve'][-1][1]:.1f}")
