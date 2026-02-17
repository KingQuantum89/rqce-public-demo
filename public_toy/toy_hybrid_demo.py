import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import os
import csv
import yaml
import numpy as np
import matplotlib.pyplot as plt

# Import YOUR existing branching module
# This uses: C:\Users\Nathan\Desktop\rqce-sims\utils\branching.py
from utils.branching import branch_expand


def load_cfg(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def measure(theta, shots, noise_p, rng):
    # Toy "quantum-like" measurement
    ideal = np.cos(theta)
    ideal = (1 - noise_p) * ideal
    p_plus = (1 + ideal) / 2
    plus = rng.binomial(shots, p_plus)
    z = (plus - (shots - plus)) / shots
    return z


def objective(z):
    return float(np.mean(z))


def update_theta(theta, z, lr):
    grad = np.sin(theta)
    return theta - lr * grad


def main():
    cfg = load_cfg("demo.yaml")
    rng = np.random.default_rng(cfg["seed"])

    outdir = os.path.join("..", cfg["outdir"])
    ensure_dir(outdir)

    # Initial branch
    branches = [{
        "id": "B0",
        "theta": rng.uniform(-np.pi, np.pi, cfg["n_qubits"]),
        "depth": 0
    }]

    log_path = os.path.join(outdir, "toy_log.csv")
    hist_obj = []
    hist_branches = []

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "branch_id", "objective", "num_branches"])

        step = 0
        for p in range(cfg["passes"]):
            for s in range(cfg["steps_per_pass"]):
                # Evaluate branches
                scores = []
                for b in branches:
                    z = measure(b["theta"], cfg["shots"], cfg["noise_p"], rng)
                    obj = objective(z)
                    scores.append((obj, b))

                # Pick best
                scores.sort(key=lambda x: x[0], reverse=True)
                obj, chosen = scores[0]

                # Update chosen branch
                z = measure(chosen["theta"], cfg["shots"], cfg["noise_p"], rng)
                chosen["theta"] = update_theta(chosen["theta"], z, cfg["lr"])

                # Branch occasionally
                if step % 10 == 0:
                    new = []
                    for k in range(cfg["branch_factor"]):
                        jitter = rng.normal(0, cfg["jitter_scale"], cfg["n_qubits"])
                        new.append({
                            "id": f"{chosen['id']}.{k}",
                            "theta": chosen["theta"] + jitter,
                            "depth": chosen["depth"] + 1
                        })
                    branches = branch_expand(branches, new_states=new)

                    # Prune
                    branches = branches[:cfg["max_branches"]]

                hist_obj.append(obj)
                hist_branches.append(len(branches))

                writer.writerow([step, chosen["id"], obj, len(branches)])
                step += 1

    # Plot
    plt.figure()
    plt.plot(hist_obj)
    plt.title("Toy Hybrid Objective")
    plt.xlabel("Step")
    plt.ylabel("Mean <Z>")
    plt.savefig(os.path.join(outdir, "toy_objective.png"), dpi=200)

    plt.figure()
    plt.plot(hist_branches)
    plt.title("Branch Count")
    plt.xlabel("Step")
    plt.ylabel("Branches")
    plt.savefig(os.path.join(outdir, "toy_branches.png"), dpi=200)

    print("DONE")
    print("CSV:", log_path)
    print("Plots saved in:", outdir)


if __name__ == "__main__":
    main()
