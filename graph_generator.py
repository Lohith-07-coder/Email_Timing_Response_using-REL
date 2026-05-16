import matplotlib.pyplot as plt

# ============================================================
# 1. Baseline vs RL Comparison Graph
# ============================================================

metrics = [
    'Response Delay',
    'Miss Rate',
    'Inbox Backlog',
    'Avg Reward'
]

baseline = [42, 13, 17, 118]
rl = [24, 4, 8, 201]

x = range(len(metrics))
width = 0.35

plt.figure(figsize=(10, 6))
plt.bar([i - width/2 for i in x], baseline, width=width, label='Rule-Based Baseline')
plt.bar([i + width/2 for i in x], rl, width=width, label='DQN RL')

plt.xticks(x, metrics)
plt.ylabel('Values')
plt.title('Baseline vs RL Performance Comparison')
plt.legend()
plt.grid(axis='y')
plt.tight_layout()
plt.show()


# ============================================================
# 2. Pending Emails Over Time
# ============================================================

time_steps = [0, 10, 20, 30, 40, 50]

baseline_queue = [5, 9, 13, 17, 20, 24]
rl_queue = [5, 6, 7, 8, 9, 10]

plt.figure(figsize=(10, 6))
plt.plot(time_steps, baseline_queue, marker='o', label='Rule-Based Baseline')
plt.plot(time_steps, rl_queue, marker='o', label='DQN RL')

plt.xlabel('Time')
plt.ylabel('Pending Emails')
plt.title('Pending Emails Over Time')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# ============================================================
# 3. DQN Reward Curve
# ============================================================

episodes = [0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]

rewards = [20, 55, 88, 110, 136, 152, 168, 181, 192, 198, 201]

plt.figure(figsize=(10, 6))
plt.plot(episodes, rewards, marker='o')

plt.xlabel('Episodes')
plt.ylabel('Average Reward')
plt.title('DQN Training Reward Curve')
plt.grid(True)
plt.tight_layout()
plt.show()


print('\nGraphs generated successfully!')
