import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
from model import QAgent
import torch
import torch.optim as optim
import gymnasium as gym
import random
import os


torch.set_num_threads(1)
os.environ["OMP_NUM_THREADS"] = "1"


def run_experiment(lr, update_frequency, network_size, gamma, epsilon, seed, n_steps=10**6):
    device = torch.device("cpu")

    # Set random seed
    torch.manual_seed(seed)
    random.seed(seed)

    env = gym.make("CartPole-v1")
    eval_env = gym.make("CartPole-v1")
    env.action_space.seed(seed)

    agent = QAgent(hidden_size=network_size).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    trained_agent, learning_curve = train_agent(env, eval_env, agent, optimizer, criterion, gamma, epsilon, n_steps, update_frequency)

    return {
        "lr": lr, "update_freq": update_frequency, "network_size": network_size,
        "gamma": gamma, "epsilon": epsilon, "seed": seed, "curve": learning_curve
    }


def train_agent(env, eval_env, agent, optimizer, criterion, gamma, epsilon, n_steps, update_frequency, device="cpu"):
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32).to(device)

    optimizer.zero_grad()

    learning_curve = []
    eval_freq = 5000

    for step in range(n_steps):
        # Predict Q-values for the current state
        predicted_q_values = agent(state)

        # Epsilon-greedy action selection
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = torch.argmax(predicted_q_values).item()


        # Take the action
        next_state, reward, done, truncated, _ = env.step(action)
        next_state = torch.tensor(next_state, dtype=torch.float32).to(device)

        if done:
            target = float(reward)
        else:
            # Predict Q-values for the next state and compute target
            # without gradients
            with torch.no_grad():
                q_values = agent(next_state)
                next_action = torch.argmax(q_values).item()

            target = reward + gamma * q_values[next_action].item()

        # Compute loss and update agent
        loss = criterion(predicted_q_values[action], torch.tensor(target, dtype=torch.float32).to(device))

        # Accumulate gradients and update every `update_frequency` steps
        loss /= update_frequency
        loss.backward()

        if (step + 1) % update_frequency == 0:
            optimizer.step()
            optimizer.zero_grad()

        state = next_state
        if done or truncated:
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float32).to(device)

        # Evaluate the agent every eval_freq steps
        if (step + 1) % eval_freq == 0:
            score = evaluate_agent(agent, eval_env)
            learning_curve.append((step, score))

    return agent, learning_curve


def evaluate_agent(agent, env, device="cpu", episodes=10):
    total_rewards = []

    for episode in range(episodes):
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32).to(device)
        sum_reward = 0

        for step in range(500):
            with torch.no_grad():
                q_values = agent(state)
            action = torch.argmax(q_values).item()
            next_state, reward, done, truncated, info = env.step(action)

            if done or truncated:
                break

            sum_reward += reward
            state = torch.tensor(next_state, dtype=torch.float32).to(device)

        total_rewards.append(sum_reward)

    return sum(total_rewards) / episodes


if __name__ == "__main__":
    # Hyperparameter ranges
    learning_rates = [0.0001, 0.001, 0.01, 0.1]
    gammas = [0.8, 0.9, 0.99]
    epsilons = [0.05, 0.1, 0.2]
    network_sizes = [64, 128, 256]
    update_frequencies = [1, 4, 8]

    # across 5 seeds
    seeds = [67, 69, 420, 666, 69420]

    all_experiments = list(itertools.product(learning_rates, update_frequencies, network_sizes, gammas, epsilons, seeds))

    completed_experiments = set()
    csv_file = "results.csv"
    if os.path.exists(csv_file):
        existing_df = pd.read_csv(csv_file)
        columns = ['Learning Rate', 'Update Frequency', 'Network Size', 'Gamma', 'Epsilon', 'Seed']

        for _, row in existing_df[columns].drop_duplicates().iterrows():
            completed_experiments.add((
                row["Learning Rate"],
                row["Update Frequency"],
                int(row["Network Size"]),
                row["Gamma"],
                row["Epsilon"],
                int(row["Seed"])
            ))


    experiments = [exp for exp in all_experiments if exp not in completed_experiments]

    print(f"Running {len(experiments)} experiments...")
    print(f"Total configurations: {len(all_experiments)}")
    print(f"Already completed: {len(completed_experiments)}")
    print(f"Running {len(experiments)} remaining experiments...")

    # Parallelisation
    results = []
    with ProcessPoolExecutor(max_workers=100) as executor:
        futures = [executor.submit(run_experiment, *exp) for exp in experiments]

        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            results.append(result)
            print(f"[{i+1}/{len(experiments)}] Finished seed {result['seed']} for config (lr={result['lr']}, net={result['network_size']}) -> Score: {result['curve'][-1][1]}")

            # Collect results into a df
            data = []
            for result in results:
                for step, score in result["curve"]:
                    data.append({
                        "Learning Rate": result["lr"],
                        "Update Frequency": result["update_freq"],
                        "Network Size": result["network_size"],
                        "Gamma": result["gamma"],
                        "Epsilon": result["epsilon"],
                        "Seed": result["seed"],
                        "Step": step,
                        "Score": score
                    })

            df = pd.DataFrame(data)

            # Save it for further use
            file_exists = os.path.exists(csv_file)
            df.to_csv(csv_file, mode='a', header=not file_exists, index=False)

            print(f"[{i + 1}/{len(experiments)}] Finished seed {result['seed']} for config "
                  f"(lr={result['lr']}, update={result['update_freq']}, net={result['network_size']}, "
                  f"gamma={result['gamma']}, eps={result['epsilon']}) -> Score: {result['curve'][-1][1]}")