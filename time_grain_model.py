# time_grain_model.py
# Minimal reproducible implementation for the paper
# "Temporal grain as a control parameter for viable recurrent computation"
# Compatible with Python 3.11, NumPy >= 1.26, SciPy >= 1.12

import numpy as np

# ---------------------------
# 1. Model parameters
# ---------------------------
N = 64                 # number of neural-mass units
tau_m = 0.015          # membrane time constant (s)
tau_d = 0.050          # conduction delay (s)
Tint = 0.200           # intrinsic integration timescale (s)
sigma = 0.05           # noise amplitude
c_E = 1.0              # energy cost coefficient (activity)
c_D = 0.2              # energy cost coefficient (derivative)
spectral_radius = 0.9  # spectral scaling of W

# ---------------------------
# 2. Helper functions
# ---------------------------
def initialize_network(N=64, density=0.2, spectral_radius=0.9, seed=None):
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((N, N)) * (rng.random((N, N)) < density)
    eigs = np.linalg.eigvals(W)
    W *= spectral_radius / np.max(np.abs(eigs))
    return W

def tanh(x):
    return np.tanh(x)

# ---------------------------
# 3. Simulation function
# ---------------------------
def simulate(W, DeltaT, T_total=40.0, Tint=0.2, sigma=0.05, seed=None):
    rng = np.random.default_rng(seed)
    dt = DeltaT
    steps = int(T_total / dt)
    delay_steps = int(tau_d / dt)
    x = np.zeros((N, steps))

    for t in range(delay_steps, steps - 1):
        delayed = x[:, t - delay_steps]
        noise = rng.normal(0, sigma, N)
        dx = (1 / tau_m) * (-x[:, t] + W @ tanh(delayed) + noise)
        x[:, t + 1] = x[:, t] + dt * dx
        x[:, t + 1] = np.clip(x[:, t + 1], -10, 10)
    
    return x[:, int(10 / dt):]  # discard first 10 s transient

# ---------------------------
# 4. Analysis metrics
# ---------------------------
def compute_temporal_integration(x, Tint, dt):
    lag = int(Tint / dt)
    ac = np.mean([np.corrcoef(xi[:-lag], xi[lag:])[0, 1] for xi in x])
    return ac

def compute_dimensionality(x):
    cov = np.cov(x)
    eigvals = np.sort(np.linalg.eigvalsh(cov))[::-1]
    cumsum = np.cumsum(eigvals) / np.sum(eigvals)
    k90 = np.searchsorted(cumsum, 0.9) + 1
    return k90 / x.shape[0]

def compute_metabolic_load(x, dt, c_E=1.0, c_D=0.2):
    dx = np.diff(x) / dt
    P = c_E * np.abs(x[:, 1:]) + c_D * np.abs(dx)
    return np.mean(np.mean(P, axis=1))

# ---------------------------
# 5. Example usage
# ---------------------------
if __name__ == "__main__":
    G_values = [0.01, 0.30, 1.00]
    results = []

    for G in G_values:
        DeltaT = G * Tint
        W = initialize_network(N, seed=42)
        x = simulate(W, DeltaT, Tint=Tint, sigma=sigma, seed=1)
        I_T = compute_temporal_integration(x, Tint, DeltaT)
        D = compute_dimensionality(x)
        P = compute_metabolic_load(x, DeltaT)
        results.append((G, I_T, D, P))

    for G, I_T, D, P in results:
        print(f"G={G:.2f}: I_T={I_T:.3f}, D={D:.3f}, P_tot={P:.3f}")
