import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# ===============================
# Toy objective function
# ===============================

def objective(x):
    # simple rugged landscape
    return np.sin(x) + 0.1 * np.random.randn()

# ===============================
# Baseline methods
# ===============================

def run_classical(n_steps=600):
    x = 0.0
    lr = 0.01
    hist = []

    for _ in range(n_steps):
        grad = np.cos(x)
        x += lr * grad
        hist.append(objective(x))

    return np.array(hist)


def run_random_search(n_steps=600):
    x = 0.0
    hist = []
    for _ in range(n_steps):
        x += np.random.randn() * 0.05
        hist.append(objective(x))
    return np.array(hist)

# ===============================
# Hybrid supervisory toy model
# ===============================

def run_hybrid_supervisor(n_steps=600, max_branches=8):
    x = 0.0
    lr = 0.01
    branches = 3
    scale = 0.05
    hist = []
    branch_hist = []

    prev = objective(x)

    stall_eps = 1e-4

    for step in range(n_steps):
        # generate branches
        candidates = x + np.random.randn(branches) * scale
        scores = np.array([objective(c) for c in candidates])
        best_idx = np.argmax(scores)
        x = candidates[best_idx]

        curr = scores[best_idx]
        improvement = curr - prev
        prev = curr

        # supervisory control logic
        if improvement < stall_eps and branches < max_branches:
            # expand exploration
            branches += 1
            scale *= 1.05

        elif improvement > 10 * stall_eps and branches > 3:
            # contract when converging
            branches -= 1
            scale *= 0.97

        hist.append(curr)
        branch_hist.append(branches)

    return np.array(hist), np.array(branch_hist)

# ===============================
# Run experiments
# ===============================

steps = 600
classical = run_classical(steps)
random_search = run_random_search(steps)
hybrid, branch_hist = run_hybrid_supervisor(steps)

# ===============================
# PLOTS
# ===============================

# Objective comparison
plt.figure(figsize=(10,5))
plt.plot(classical, label="Classical Gradient")
plt.plot(random_search, label="Random Search")
plt.plot(hybrid, label="Hybrid Supervisor")
plt.xlabel("Step")
plt.ylabel("Objective Value")
plt.title("Toy Hybrid Benchmark Objective")
plt.legend()
plt.tight_layout()
plt.savefig("toy_baseline_objective.png")
plt.show()

# Branch evolution
plt.figure(figsize=(10,5))
plt.plot(branch_hist)
plt.xlabel("Step")
plt.ylabel("Branches")
plt.title("Hybrid Branch Count (Adaptive)")
plt.tight_layout()
plt.savefig("toy_branch_count.png")
plt.show()

# ===============================
# Save logs
# ===============================

np.savetxt("toy_hybrid_log.csv", hybrid, delimiter=",")
np.savetxt("toy_branch_log.csv", branch_hist, delimiter=",")
np.savetxt("toy_classical_log.csv", classical, delimiter=",")
np.savetxt("toy_random_log.csv", random_search, delimiter=",")
