import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


df = pd.read_csv("results.csv")
given_baseline_df = pd.read_csv('BaselineDataCartPole.csv')

hyperparameters = ['Learning Rate', 'Update Frequency', 'Network Size', 'Gamma', 'Epsilon']

# Smooth the scores using a rolling average for better visualization
df['Score_smooth'] = df.groupby(hyperparameters + ['Seed'])['Score'].transform(
    lambda x: x.rolling(window=10, min_periods=1).mean()
)

# Find the final score for each configuration
final_eval = df['Step'].max()
final_scores_df = df[df['Step'] == final_eval]
final_scores = final_scores_df.groupby(hyperparameters)['Score'].mean().reset_index()

# Choose the highest score as baseline
best_config_id = final_scores['Score'].idxmax()
best_config = final_scores.loc[best_config_id]

best_params = {param: best_config[param] for param in hyperparameters}

# Show the best configuration
for param, value in best_params.items():
    print(f"{param}: {value}")
print(f"Final Score: {best_config['Score']}")

# Make plots for each parameter
for parameter in hyperparameters:
    remaining_params = [p for p in hyperparameters if p != parameter]

    # Filter the df for the baseline configuration except for the current parameter
    query_str = " and ".join([f"`{p}` == {best_params[p]}" for p in remaining_params])
    plot_df = df.query(query_str)

    fixed_params_string = ' ,'.join([f"{p}={best_params[p]}" for p in remaining_params])

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=plot_df, x="Step", y="Score_smooth", hue=parameter, palette="tab10", errorbar='sd')
    plt.title(f"Ablation: {fixed_params_string}\n Fixed at: {fixed_params_string}")
    plt.xlabel("Environment Steps")
    plt.ylabel("Average Return (across 5 seeds)")
    plt.grid(True)
    plt.savefig(f"ablation_{parameter.replace(' ', '_').lower()}.png")
    plt.close()


best_query_str = " and ".join([f"`{p}` == {best_params[p]}" for p in hyperparameters])
best_df = df.query(best_query_str)

plt.figure(figsize=(10, 6))

sns.lineplot(data=best_df, x="Step", y="Score_smooth", label='Best Found Configuration', color='blue', errorbar='sd')
sns.lineplot(data=given_baseline_df, x="env_step", y="Episode_Return_smooth", label='Provided Baseline', color='red')
sns.lineplot(data=given_baseline_df, x="env_step", y="Episode_Return", color='red', alpha=0.3, label='_nolegend_')

plt.title(f"Performance Comparison: Best Found Configuration vs Provided Baseline")
plt.xlabel("Environment Steps")
plt.ylabel("Average Return")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("baseline_comparison.png")

plt.close()