import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view
from itertools import combinations_with_replacement
from scipy.special import comb
from scipy.signal import lfilter
import time
from numba import njit
from tqdm import tqdm
import json

class VolterraBasis:
    """
    Efficiently constructs the regressor matrix for a Volterra series
    of memory M and degree P using NumPy vectorization.
    """
    def __init__(self, M, P):
        self.M = M
        self.P = P

        self.term_indices = []
        for p in range(1, P + 1):
            self.term_indices.extend(list(combinations_with_replacement(range(M), p)))

        self.D = len(self.term_indices)

    def transform(self, u):
        X_lagged = sliding_window_view(u, window_shape=self.M)
        N_samples = X_lagged.shape[0]
        N_features = len(self.term_indices)

        Phi = np.zeros((N_samples, N_features))

        for idx, lags in enumerate(self.term_indices):
            Phi[:, idx] = np.prod(X_lagged[:, lags], axis=1)

        return Phi

class NARMAXBasis:
    """
    Constructs the regressor matrix for a polynomial NARMAX model.
    Variables: y (M lags), u (M lags), w (M lags).
    Total Variables K = 3*M.
    Dimension: Binom(K + P, P)
    """
    def __init__(self, M, P):
        self.M = M
        self.P = P
        self.K = 3 * M 
        self.D = int(comb(self.K + P, P))
        
        self.term_indices = []
        for p in range(0, P + 1):
            self.term_indices.extend(list(combinations_with_replacement(range(self.K), p)))
            
    def transform(self, u, y, w):
        # Ensure lengths match
        min_len = min(len(u), len(y), len(w))
        u = u[:min_len]
        y = y[:min_len]
        w = w[:min_len]
        
        U_lag = sliding_window_view(u, window_shape=self.M)
        Y_lag = sliding_window_view(y, window_shape=self.M)
        W_lag = sliding_window_view(w, window_shape=self.M)
        
        # Combine into one big feature vector per time step
        # Shape: (N_samples, 3M)
        # Structure: [y_lags, u_lags, w_lags]
        features = np.hstack([Y_lag, U_lag, W_lag])
        
        N_samples = features.shape[0]
        N_features = len(self.term_indices)
        Phi = np.ones((N_samples, N_features))
        
        for idx, lags in enumerate(self.term_indices):
            if len(lags) > 0:
                Phi[:, idx] = np.prod(features[:, lags], axis=1)
                
        return Phi

# As we need to solve a lot of runwise least squares problems, we implement a fast RLS solver.
@njit
def fast_rls_curve(Phi, y, theta_star, initial_P_scale=1e8):
    """
    Efficiently computes the parameter error trajectory using Recursive Least Squares (RLS).
    This allows us to see the error at *every* sample size N from 1 to N_max
    in O(N*D^2) time, which is much faster than solving batch Cholesky for many N.
    
    The initial_P_scale is set to a large prior to mimic the unregularized case. Adjust if needed for stability.
    """
    N_samples, D = Phi.shape
    
    # Initialize Parameter Estimate
    theta = np.zeros(D, dtype=np.float64)
    
    # Initialize Inverse Covariance Matrix P = (1/lambda) * I
    # A large scale implies small regularization (lambda -> 0)
    P = np.eye(D, dtype=np.float64) * initial_P_scale
    
    # Pre-allocate error trajectory
    errors = np.zeros(N_samples, dtype=np.float64)
    
    for t in range(N_samples):
        # 1. Get new feature vector
        phi_t = Phi[t, :]
        
        # 2. Compute Gain Vector k = P * phi / (1 + phi.T * P * phi)
        # Note: We compute P @ phi first to reuse it
        P_phi = P @ phi_t
        gain_denom = 1.0 + np.dot(phi_t, P_phi)
        k = P_phi / gain_denom
        
        # 3. Compute A Priori Error
        pred = np.dot(phi_t, theta)
        e = y[t] - pred
        
        # 4. Update Parameter Estimate
        theta += k * e
        
        # 5. Update Covariance Matrix P
        # P_new = P - outer(k, P_phi)
        # We manually compute the outer product update
        # This is the O(D^2) step
        P -= np.outer(k, P_phi)
        
        # 6. Compute True Parameter Error for this N
        # (N = t + 1)
        diff = theta - theta_star
        errors[t] = np.sqrt(np.dot(diff, diff))
        
    return errors


# This function estimates beta using the Eigenvector method, which is more efficient than random sampling.
def estimate_metrics_eigen(system, generator_func, alpha, n_samples=40000):
    """
    Efficiently estimates beta using the Eigenvector method.
    Instead of random sampling, we find the direction of minimum variance
    (min eigenvector of Sigma) and test the probability mass along that vector.
    
    Returns:
        beta_prob: P(<v_min, phi>^2 > alpha)
        lambda_min: Minimum eigenvalue (The Spectral Beta)
        trace_metric: 1 / Trace(Sigma^-1) (The Average Case Proxy)
    """
    u_long = generator_func(n_samples + system.M)
    Phi = system.transform(u_long)
    
    # 1. Compute Covariance
    Sigma = (Phi.T @ Phi) / n_samples
    
    # 2. Eigendecomposition to find "worst" directions analytically
    evals, evecs = np.linalg.eigh(Sigma)
    
    # 3. Identify min eigenvalue and corresponding vector
    # (eigh returns sorted eigenvalues)
    lambda_min = evals[0]
    v_min = evecs[:, 0]
    
    # 4. Measure Probability Beta along this worst-case direction
    # This replaces the Monte Carlo random sampling
    projections = Phi @ v_min
    beta_prob = np.mean((projections**2) > alpha)
    
    # 5. Trace metric for average case comparison
    # Handle small evals to avoid explosion
    safe_evals = np.maximum(evals, 1e-12)
    trace_inv = np.sum(1.0 / safe_evals)
    metric_trace = 1.0 / trace_inv
    
    return beta_prob, lambda_min, metric_trace

def generate_true_system(D, sparsity=0.1):
    """Generates a random sparse theta."""
    theta = np.zeros(D)
    # Always keep bias small
    theta[0] = 0.0
    # Random indices
    nnz = max(1, int(D * sparsity))
    indices = np.random.choice(range(1, D), nnz, replace=False)
    theta[indices] = np.random.randn(nnz)
    # Normalize
    theta = theta / np.linalg.norm(theta)
    return theta

def generate_stable_narmax_theta(basis, sparsity=0.1, stability_factor=0.9):
    """
    Generates theta for NARMAX. Ensures linear stability by scaling autoregressive terms.
    """
    theta = np.zeros(basis.D)
    
    # 1. Set Linear AR part for stability (y[t] = stability_factor * y[t-1])
    # Corresponds to term (0,) in term_indices if variables are ordered [y... u... w...]
    # y[t-1] is index 0 in the feature vector.
    try:
        idx_y1 = basis.term_indices.index((0,))
        theta[idx_y1] = stability_factor # Decay
    except ValueError:
        pass 

    # 2. Add random nonlinear terms (sparse)
    nnz = max(1, int(basis.D * sparsity))
    cnt = 0
    while cnt < nnz:
        idx = np.random.randint(1, basis.D)
        # Check if this term contains purely Y lags (risky)
        term = basis.term_indices[idx]
        is_pure_y = all(x < basis.M for x in term)
        
        if not is_pure_y:
            theta[idx] = np.random.randn() * 0.1 # Small coefs
            cnt += 1
            
    return theta

def simulate_narmax(n_samples, M, theta, basis, u_in, w_in):
    """
    Python-loop simulation of NARMAX.
    """
    y = np.zeros(n_samples + M)
    u = np.zeros(n_samples + M)
    w = np.zeros(n_samples + M)
    
    # Pad inputs
    u[M:] = u_in
    w[M:] = w_in
    
    # Identify active terms to speed up
    active_indices = np.where(np.abs(theta) > 1e-9)[0]
    active_theta = theta[active_indices]
    active_terms = [basis.term_indices[i] for i in active_indices]
    
    # Variable mapping in transform: [y_lags, u_lags, w_lags]
    # y_lags for time t: [y[t-M], ..., y[t-1]]  <-- sliding_window_view output order
    # CAUTION: sliding_window_view(arr)[i] gives arr[i:i+M].
    # If we feed it to transform, it expects windows.
    # We must match the order expected by transform.
    # transform uses sliding_window_view which returns [x[t], x[t+1] ... x[t+M-1]]
    # Here we are at step t. We want to predict y[t].
    # The regressor vector corresponds to the window ending at t-1.
    # y window: y[t-M] ... y[t-1]
    
    for t in range(M, n_samples + M):
        # Construct local state vector
        # Window corresponding to what sliding_window_view produces for index t-M
        # i.e., arr[t-M : t] -> [y[t-M], ..., y[t-1]]
        
        state_y = y[t-M:t]
        state_u = u[t-M:t]
        state_w = w[t-M:t]
        
        # Combine exactly as NARMAXBasis.transform does: hstack([Y, U, W])
        state = np.concatenate([state_y, state_u, state_w])
        
        val = 0.0
        for i, term in enumerate(active_terms):
            term_val = 1.0
            for var_idx in term:
                term_val *= state[var_idx]
            val += term_val * active_theta[i]
            
        y[t] = val + w[t] # Add noise w[t]
        
        # Stability Check (Blowup guard)
        if abs(y[t]) > 1e4:
            y[t] = np.nan
            break
            
    return y[M:]

# A simple Lorenz system generator for chaotic noise, compiled with Numba for speed.
@njit
def generate_lorenz_numba(N, dt=0.01, sigma=10.0, rho=28.0, beta=8.0/3.0, scale=0.01):
    """
    A simple Lorenz system generator for chaotic time-correlated noise.
    Solved using Euler's method for simplicity.
    Compiled to machine code using LLVM.
    """
    noise = np.empty(N, dtype=np.float64)
    
    # State initialization
    x = np.random.rand() * 20.0 - 10.0
    y = np.random.rand() * 20.0 - 10.0
    z = np.random.rand() * 20.0 - 10.0
    
    # Burn-in
    for _ in range(500):
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        x += dx * dt
        y += dy * dt
        z += dz * dt

    # Main Loop
    for i in range(N):
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        
        x += dx * dt
        y += dy * dt
        z += dz * dt
        
        noise[i] = x * scale
        
    return noise

def normalize(raw, variance=1):
    mean = np.mean(raw)
    raw = raw - mean
    raw = raw / np.std(raw) * np.sqrt(variance)
    return raw + mean

def solve_cholesky(G, h, reg=1e-8):
    """
    Optimized approach: Ridge Regression + Cholesky Solve.
    1. Adds tiny regularization (reg) to diagonal to ensure Positive Definite.
    2. Uses 'assume_a="pos"' to trigger LAPACK Cholesky routine (dposv).
    
    This is O(D^3) but with a much smaller constant than SVD.
    """
    # G + reg*I
    G_reg = G.copy()
    G_reg[np.diag_indices_from(G_reg)] += reg
    
    # Scipy's solve with assume_a='pos' uses Cholesky decomposition
    return sp.linalg.solve(G_reg, h, assume_a='pos')

def generate_white_noise(n_samples, std=1):
    return np.random.normal(0, std, size=n_samples)

def generate_ar1_input(n_samples, rho=0.7):
    innovations = np.random.normal(0, 1, size=n_samples)
    filtered = lfilter([1], [1, -rho], innovations)
    return filtered / np.std(filtered)  # Normalize to unit variance

def solve_ridge(Phi, y, lam=1e-4):
    """
    Solves (Phi^T Phi + lam * I) theta = Phi^T y using Cholesky.
    """
    D = Phi.shape[1]
    G = Phi.T @ Phi
    G[np.diag_indices(D)] += lam
    h = Phi.T @ y
    return sp.linalg.solve(G, h, assume_a='pos')

def estimate_beta_eigen(system, generator_func, alpha, n_samples=20000):
    beta,_,_ = estimate_metrics_eigen(system, generator_func, alpha, n_samples) 


def run_experiment_1(n_trials = 1000, M=5, P=3):
    v_basis = VolterraBasis(M, P)
    D = len(v_basis.term_indices)
    
    print(f"Volterra System: Memory M={M}, Degree P={P}")
    print(f"Dimension of parameter space D: {D}")
    
    beta_iid, _, tr_iid = estimate_metrics_eigen(v_basis, lambda n: generate_white_noise(n, std=1), alpha=0.1, n_samples=100000)
    beta_ar1, _, tr_ar1 = estimate_metrics_eigen(v_basis, lambda n: normalize(generate_ar1_input(n, rho=0.7)), alpha=0.1, n_samples=100000)
    print(f"\nEstimated Beta (Gaussian Noise): {beta_iid:.4f}")
    print(f"Estimated Beta (AR(1) Noise): {beta_ar1:.4f}")
    
    ratios = np.logspace(-0.3, 1.7, num=1000) # From ~0.5 to ~50 on log scale 
    sample_sizes = (ratios * D).astype(int)
    
    theta_star = np.random.standard_normal(D)
    theta_star /= np.linalg.norm(theta_star)

    print(f"\nRunning {n_trials} trials (Batch Least Squares)...")
    print(f"Noise Profiles: Gaussian vs AR(1) (rho=0.7)")
   
    # Define thresholds for probability plot (log spaced from 1 to 100)
    # This checks P(Error < 1), P(Error < 3.16), ..., P(Error < 100)
    prob_thresholds = np.geomspace(1, 100, 5)

    data_export = {
            'Ratio': ratios,
            'GaussianRatioBeta': ratios*beta_iid,
            'AR1RatioBeta': ratios*beta_ar1,
            'GaussianRatioTrace': ratios*tr_iid,
            'AR1RatioTrace': ratios*tr_ar1,
    }
    
    percentiles = [5, 20, 35, 50, 65, 80, 95]

    generators = [("Gaussian", lambda n: generate_white_noise(n, std=1)),
                  ("AR1", lambda n: generate_ar1_input(n, rho=0.7)),
    ] 

    metadata = {
        'D': int(D),
        'M': int(M),
        'P': int(P),
        'Beta_Gaussian': float(beta_iid),
        'Beta_AR1': float(beta_ar1),
        'Trace_Gaussian': float(tr_iid),
        'Trace_AR1': float(tr_ar1),
        'ProbThresholds': prob_thresholds.tolist()  # Convert numpy array to list
    }
    
    meta_filename = 'experiment1_metadata.json'
    with open(meta_filename, 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f"Metadata successfully saved to {meta_filename}")

    
    # This allows \input{experiment1_vars.tex} and using \ExpOneBetaGauss in TikZ
    tex_filename = 'experiment1_vars.tex'
    with open(tex_filename, 'w') as f:
        f.write("% Auto-generated variables from Experiment 1\n")
        f.write(f"\\newcommand{{\\ExpOneD}}{{{int(D)}}}\n")
        f.write(f"\\newcommand{{\\ExpOneM}}{{{int(M)}}}\n")
        f.write(f"\\newcommand{{\\ExpOneP}}{{{int(P)}}}\n")
        f.write(f"\\newcommand{{\\ExpOneBetaGauss}}{{{beta_iid:.4f}}}\n")
        f.write(f"\\newcommand{{\\ExpOneBetaAR}}{{{beta_ar1:.4f}}}\n")
        f.write(f"\\newcommand{{\\ExpOneTraceGauss}}{{{tr_iid:.4f}}}\n")
        f.write(f"\\newcommand{{\\ExpOneTraceAR}}{{{tr_ar1:.4f}}}\n")
        # Save thresholds list for potentially looping in TikZ
        thresh_str = ",".join([f"{t:.2f}" for t in prob_thresholds])
        f.write(f"\\newcommand{{\\ExpOneProbThresholds}}{{{thresh_str}}}\n")
    print(f"LaTeX variables successfully saved to {tex_filename}")



    for noise_type, generator in generators:
        errors = np.zeros((len(sample_sizes), n_trials))
        for trial in tqdm(range(n_trials), desc=f"{noise_type} Trials", unit="trial"):
            u_full = generator(sample_sizes[-1] + M - 1)
            Phi_full = v_basis.transform(u_full)
            y_full = Phi_full @ theta_star + generate_white_noise(sample_sizes[-1])  # Add noise

            traj_errors = fast_rls_curve(Phi_full, y_full, theta_star, initial_P_scale=1e8)
            
            # Extract errors at the specific sample sizes we care about
            # sample_sizes are 1-based counts. Index is N-1.
            indices = sample_sizes - 1
            errors[:, trial] = traj_errors[indices]

            #for i, N in enumerate(sample_sizes):
            #    Phi_N = Phi_full[:N, :]
            #    y_N = y_full[:N]
            #    theta_hat = solve_cholesky(Phi_N.T @ Phi_N, Phi_N.T @ y_N)
            #    errors[i, trial] = np.linalg.norm(theta_hat - theta_star)
        
        
        median_err = np.median(errors, axis=1)
        plt.plot(ratios, median_err, label=f'{noise_type} Noise')
        p05 = np.percentile(errors, 5, axis=1)
        p95 = np.percentile(errors, 95, axis=1) 

        plt.fill_between(sample_sizes / D, p05, p95, alpha=0.2)

        p_vals = np.percentile(errors, percentiles, axis=1)
        
        # Store in dictionary (e.g., Gaussian_P05, Gaussian_P50)
        for p, val in zip(percentiles, p_vals):
            col_name = f"{noise_type}_P{p:02d}"
            data_export[col_name] = val

        # Compute Probabilities for thresholds
        for T in prob_thresholds:
            # Probability that error is smaller than T
            probs = np.mean(errors < T, axis=1)
            col_name = f"{noise_type}_Prob_Lt_{T:.2f}"
            data_export[col_name] = probs
    
    # Create DataFrame and Save
    df = pd.DataFrame(data_export)
    filename = 'experiment1.csv'
    df.to_csv(filename, index=False, float_format='%.6f')
    print(f"\nData successfully saved to {filename}")
    print(df.head())
        
    plt.axvline(1.0, color='r', linestyle='--', label='Theoretical Threshold (N=D)')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'Sample Complexity Ratio $N / D$')
    plt.ylabel(r'Parameter Error $\|\hat{\theta}_N - \theta^\star \|_2$')
    plt.title(f'Parameter Estimation \nNoise Comparison')
    plt.grid(True, which="both", ls="-")
    plt.legend()
    plt.savefig('experiment1_noise_comparison.png')

    plt.figure()
    for (excitation, _) in generators:
        plt.plot(data_export[f'{excitation}RatioBeta'], data_export[f'{excitation}_P50'], label=f'{excitation} Median (Beta)')
        plt.fill_between(data_export[f'{excitation}RatioBeta'], data_export[f'{excitation}_P05'], data_export[f'{excitation}_P95'], alpha=0.2, label=f'{excitation} 5-95% (Beta)')
        
        plt.plot(data_export[f'{excitation}RatioTrace'], data_export[f'{excitation}_P50'], label=f'{excitation} Median (Trace)')
        plt.fill_between(data_export[f'{excitation}RatioTrace'], data_export[f'{excitation}_P05'], data_export[f'{excitation}_P95'], alpha=0.2, label=f'{excitation} 5-95% (Trace)')


        
    
    plt.axvline(1.0, color='r', linestyle='--', label='Theoretical Threshold (N=D)')
    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel(r'Parameter Error $\|\hat{\theta}_N - \theta^\star \|_2$')
    plt.title(f'Parameter Estimation \nNoise Comparison')
    plt.grid(True, which="both", ls="-")
    plt.legend()

    # New Plot: Probability of Convergence
    plt.figure(figsize=(10, 6))
    # Generate colors for thresholds
    colors = plt.cm.viridis(np.linspace(0, 1, len(prob_thresholds)))
    
    for i, T in enumerate(prob_thresholds):
        # Gaussian (Solid)
        col_g = f"Gaussian_Prob_Lt_{T:.2f}"
        plt.plot(ratios, data_export[col_g], color=colors[i], linestyle='-', label=f'Gaussian < {T:.1f}')
        
        # AR1 (Dashed)
        col_a = f"AR1_Prob_Lt_{T:.2f}"
        plt.plot(ratios, data_export[col_a], color=colors[i], linestyle='--', label=f'AR1 < {T:.1f}')
        
    plt.axvline(1.0, color='r', linestyle=':', label='N=D')
    plt.xscale('log')
    plt.xlabel(r'Sample Complexity Ratio $N / D$')
    plt.ylabel('Probability')
    plt.title(f'Convergence Probability P(Error < Threshold)')
    plt.grid(True, which="both", ls="-")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('experiment1_convergence_prob.png')
    
def run_experiment_2(n_trials = 1000, M=5, P=2):
    print("\n--- Experiment 2: Noise Robustness ---")
    basis = VolterraBasis(M, P)
    D = len(basis.term_indices)

    #ratios = np.array([0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0])
    ratios = np.logspace(-0.3, 1.7, num=1000) # From ~0.5 to ~50 on log scale 
    sample_sizes = (ratios * D).astype(int)

    theta_star = generate_true_system(D)
    
    results = {'N': sample_sizes}
    
    percentiles = [5, 20, 35, 50, 65, 80, 95]

    # Noise generators
    noises = {
        'Gaussian': lambda n: np.random.normal(0, 1, n),
        'Uniform': lambda n: normalize(np.random.rand(n)-0.5), # Unit variance
        'AR(1)': lambda n: normalize(generate_ar1_input(n, rho=0.7)), # Unit variance
        'Lorentz': lambda n: normalize(generate_lorenz_numba(n) + generate_white_noise(n,0.01)) #Unit variance
    } 
    
    plt.figure()
    for name, noise_gen in noises.items():
        errors = np.zeros((len(sample_sizes), n_trials))
        for trial in tqdm(range(n_trials), desc=f"{name} Trials", unit="trial"):
            u_full = generate_white_noise(sample_sizes[-1] + M - 1)
            Phi_full = basis.transform(u_full)
            y_full = Phi_full @ theta_star + noise_gen(sample_sizes[-1])  # Add noise

            for i, N in enumerate(sample_sizes):
                Phi_N = Phi_full[:N, :]
                y_N = y_full[:N]
                theta_hat = solve_cholesky(Phi_N.T @ Phi_N, Phi_N.T @ y_N)
                errors[i, trial] = np.linalg.norm(theta_hat - theta_star)

        median_err = np.median(errors, axis=1)
        plt.plot(ratios, median_err, label=f'{name} Noise')
        p05 = np.percentile(errors, 5, axis=1)
        p95 = np.percentile(errors, 95, axis=1) 

        plt.fill_between(sample_sizes / D, p05, p95, alpha=0.2)

        p_vals = np.percentile(errors, percentiles, axis=1)
        
        # Store in dictionary (e.g., Gaussian_P05, Gaussian_P50)
        for p, val in zip(percentiles, p_vals):
            col_name = f"{name}_P{p:02d}"
            results[col_name] = val
        
                
    df = pd.DataFrame(results)
    df.to_csv('experiment2.csv', index=False)
    
    plt.xscale('log') 
    plt.yscale('log')
    plt.xlabel('N')
    plt.ylabel('Error')
    plt.legend()
    plt.title('Experiment 2: Noise Robustness')
    plt.savefig('experiment2_noise.png')

    noise_results = dict()

    fig, ax = plt.subplots(len(noises), 1, figsize=(10, 12))
    # Plot some sample trajectories of the noise processes to visualize their differences
    idx = 0
    for name, noise_gen in noises.items():
        for _ in range(5): # Plot 5 trajectories for each noise type
            sample_noise = noise_gen(1000)
            noise_results[f'{name}_{idx}'] = sample_noise 
            ax[idx].plot(sample_noise, label=f'{name} Noise')
        idx += 1
    plt.title('Sample Noise Trajectories')
    plt.xlabel('Time Steps')
    plt.ylabel('Noise Value')
    plt.legend()
    plt.grid(True)
    plt.savefig('experiment2_noise_trajectories.png')

    df = pd.DataFrame(noise_results)
    df.to_csv('experiment2_noise_trajectories.csv', index=False)


def generate_and_save_noise_trajectories(n_samples=1000, M=5):
    """Generates M noise trajectories of length n_samples for each noise type."""
    # Noise generators
    noises = {
        'Gaussian': lambda n: np.random.normal(0, 1, n),
        'Uniform': lambda n: normalize(np.random.rand(n)-0.5), # Unit variance
        'AR(1)': lambda n: normalize(generate_ar1_input(n, rho=0.7)), # Unit variance
        'Lorentz': lambda n: normalize(generate_lorenz_numba(n) + generate_white_noise(n,0.01)) #Unit variance
    }

    noise_results = dict()
    for name, noise_gen in noises.items():
        for i in range(M):
            noise_results[f'{name}_{i}'] =noise_gen(n_samples)
    
    df = pd.DataFrame(noise_results)
    df.to_csv('experiment2_noise_trajectories.csv', index=False)


def run_experiment_3(n_trials=1000):
    print("\n--- Experiment 3: Curse of Dimensionality ---")
    # We want to find N required to reach error < 0.1
    target_error = 1
    configs = [(5, 2), (5, 3), (5, 4), (5,5), (10, 2), (10, 3)]
    
    D_vals = []
    N_req_vals = []
    
    for M, P in configs:
        basis = VolterraBasis(M, P)
        D = basis.D
        D_vals.append(D)
        
        sample_size = 10 * D # We checked previously that 2*D was enough, so 4D should be save.
        errors = np.zeros((sample_size, n_trials))
        theta_star = generate_true_system(D)

        for trial in tqdm(range(n_trials), desc=f"Volterra({M},{P}) Trials", unit="trial"):
            u_full = generate_white_noise(sample_size + M - 1)
            Phi_full = basis.transform(u_full)
            y_full = Phi_full @ theta_star + generate_white_noise(sample_size)  # Add noise

            traj_errors = fast_rls_curve(Phi_full, y_full, theta_star, initial_P_scale=1e8)
            
            # Extract errors at the specific sample sizes we care about
            # sample_sizes are 1-based counts. Index is N-1.
            errors[:, trial] = traj_errors

            #for i, N in enumerate(sample_sizes):
            #    Phi_N = Phi_full[:N, :]
            #    y_N = y_full[:N]
            #    theta_hat = solve_cholesky(Phi_N.T @ Phi_N, Phi_N.T @ y_N)
            #    errors[i, trial] = np.linalg.norm(theta_hat - theta_star)
       
        mean_err = np.mean(errors, axis=1)
        
        # Find all indices where the error is still too high
        too_high_indices = np.flatnonzero(mean_err >= target_error)

        if too_high_indices.size == 0:
            # Every single n (1 to N) satisfies the condition
            found_N = 1
        elif too_high_indices[-1] == len(mean_err) - 1:
            # Even the largest n (N) is still above the target error
            found_N = None
        else:
            # The smallest N is the index after the last failure + 1 (for 1-based indexing)
            # Index 'i' corresponds to sample size 'i + 1', so the next is 'i + 2'
            found_N = too_high_indices[-1] + 2        

        N_req_vals.append(found_N)
        print(f"M={M}, P={P} -> D={D}, Req N={found_N}")

    plt.figure()
    plt.loglog(D_vals, N_req_vals, 'ko-', label='Empirical')
    # Plot Reference line y=x
    plt.plot(D_vals, D_vals, 'r--', label='Linear Scaling')
    plt.xlabel('Dimension D')
    plt.ylabel('Required N')
    plt.title('Experiment 3: Curse of Dimensionality')
    plt.legend()
    plt.grid(True)
    plt.savefig('experiment3_curse.png')
    
    pd.DataFrame({'D': D_vals, 'N_req': N_req_vals}).to_csv('experiment3.csv', index=False)

def run_experiment_4():
    print("\n--- Experiment 4: Cost of Feedback (Volterra vs NARMAX) ---")
    M, P = 3, 2 # Small system to keep NARMAX dim manageable
    
    # 1. Volterra
    volt_basis = VolterraBasis(M, P)
    # 2. NARMAX
    narmax_basis = NARMAXBasis(M, P)
    
    print(f"Volterra D={volt_basis.D}, NARMAX D={narmax_basis.D}")
    
    ratios = np.array([2, 5, 10, 20, 50])
    
    res_volt = []
    res_narmax = []
    
    # Run Volterra (Generate True Volterra -> Estimate Volterra)
    theta_v = generate_true_system(volt_basis.D)
    for r in tqdm(ratios, desc='Volterra'):
        N = int(r * volt_basis.D)
        u = np.random.normal(0, 1, N+M)
        Phi = volt_basis.transform(u)[:N]
        y = Phi @ theta_v + np.random.normal(0, 0.1, N)
        th = solve_ridge(Phi, y)
        res_volt.append(np.linalg.norm(th - theta_v))
        
    # Run NARMAX (Generate True NARMAX -> Estimate NARMAX)
    # Oracle assumption: we construct regressor from True Y, True U, True W
    theta_n = generate_stable_narmax_theta(narmax_basis, sparsity=0.05, stability_factor=0.6)
    
    for r in tqdm(ratios, desc='NARMAX'):
        N = int(r * narmax_basis.D)
        
        # Simulate System
        u_in = np.random.normal(0, 1, N)
        w_in = np.random.normal(0, 0.1, N)
        
        # Burn in
        y_out = simulate_narmax(N, M, theta_n, narmax_basis, u_in, w_in)
        
        # Check for stability failure
        if np.isnan(y_out).any():
            res_narmax.append(np.nan)
            continue
            
        # Construct Regressor (Oracle)
        # Pad inputs to match transform expectation
        # transform expects full history.
        u_full = np.concatenate([np.zeros(M), u_in])
        w_full = np.concatenate([np.zeros(M), w_in])
        y_full = np.concatenate([np.zeros(M), y_out])
        
        # FIX: Slice to N because transform produces N+1 rows for input length N+M
        Phi = narmax_basis.transform(u_full, y_full, w_full)[:N]
        
        th = solve_ridge(Phi, y_out, lam=1e-4) # Higher Reg for NARMAX
        res_narmax.append(np.linalg.norm(th - theta_n))

    plt.figure()
    plt.plot(ratios, res_volt, 'b-o', label='Volterra')
    plt.plot(ratios, res_narmax, 'r-s', label='NARMAX')
    plt.xlabel('Ratio N/D')
    plt.ylabel('Error')
    plt.yscale('log')
    plt.legend()
    plt.title('Experiment 4: Cost of Feedback')
    plt.savefig('experiment4_feedback.png')
    
    pd.DataFrame({'Ratio': ratios, 'Volterra': res_volt, 'NARMAX': res_narmax}).to_csv('experiment4.csv', index=False)

def run_experiment_5():
    print("\n--- Experiment 5: Stability & Regularization ---")
    M, P = 2, 2
    basis = NARMAXBasis(M, P)
    D = basis.D
    N = 1000
    
    # 1. Stable System
    theta_stable = generate_stable_narmax_theta(basis, stability_factor=0.5)
    # 2. Near Unstable
    theta_unstable = generate_stable_narmax_theta(basis, stability_factor=0.99)
    
    lambdas = np.logspace(-6, 6, 30)
    
    err_stable = []
    err_unstable = []
    norm_stable = []
    norm_unstable = []
    
    for th, err_list, norm_list in [(theta_stable, err_stable, norm_stable), 
                                    (theta_unstable, err_unstable, norm_unstable)]: 
        # Generate Data ONCE
        u_in = np.random.normal(0, 1, N)
        w_in = np.random.normal(0, 0.1, N)
        y_out = simulate_narmax(N, M, th, basis, u_in, w_in)
        
        if np.isnan(y_out).any():
            print("System unstable during generation.")
            # Fill with NaNs
            err_list.extend([np.nan]*len(lambdas))
            norm_list.extend([np.nan]*len(lambdas))
            continue
            
        u_full = np.concatenate([np.zeros(M), u_in])
        w_full = np.concatenate([np.zeros(M), w_in])
        y_full = np.concatenate([np.zeros(M), y_out])
        
        # FIX: Slice to N
        Phi = basis.transform(u_full, y_full, w_full)[:N]
        
        # Check Regressor Norm
        avg_norm = np.mean(np.linalg.norm(Phi, axis=1))
        
        for lam in lambdas:
            theta_hat = solve_ridge(Phi, y_out, lam=lam)
            err_list.append(np.linalg.norm(theta_hat - th))
            norm_list.append(avg_norm)
            
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].loglog(lambdas, err_stable, 'b-o', label='Stable (0.5)')
    ax[0].loglog(lambdas, err_unstable, 'r-s', label='Near Unstable (0.99)')
    ax[0].set_xlabel('Lambda')
    ax[0].set_ylabel('Error')
    ax[0].set_title('Regularization Path')
    ax[0].legend()
    
    # Plot Regressor Norms (Constant vs Lambda, but different between systems)
    ax[1].bar(['Stable', 'Unstable'], [np.nanmean(norm_stable), np.nanmean(norm_unstable)])
    ax[1].set_title('Average Regressor Norm')
    ax[1].set_yscale('log')
    
    plt.savefig('experiment5_stability.png')
    
    pd.DataFrame({
        'Lambda': lambdas, 
        'Err_Stable': err_stable, 
        'Err_Unstable': err_unstable
    }).to_csv('experiment5.csv', index=False)

def main():
    # Run all experiments
    #run_experiment_1(M=5, P=4)
    #run_experiment_2()
    #generate_and_save_noise_trajectories()
    run_experiment_3()
    #run_experiment_4()
    #run_experiment_5()
    print("\nAll experiments completed.")
    
    plt.show()

if __name__ == "__main__":
    # Fix the random seed for reproducibility across all experiments
    np.random.seed(0)

    main()
