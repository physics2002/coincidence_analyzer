import numpy as np
from scipy.optimize import curve_fit

def exponential_decay(t, A, lambd, B):
    return A * np.exp(-lambd * t) + B

def exponential_decay_no_bg(t, A, lambd):
    return A * np.exp(-lambd * t)

def fit_decay_curve(signal_dict, bg_dict=None, half_life=None, time_window=None, no_background=False):
    """
    Fit count rate vs time data to exponential decay model.
    Set no_background=True to fit without a background term.
    If interactive_time_window=True, user selects time window on a plot.
    """
    t = signal_dict['centers']
    y = signal_dict['rates']
    yerr = signal_dict['errors']

    # Apply time window filter
    if time_window:
        t_min, t_max = time_window
        mask = (t >= t_min) & (t <= t_max)
        t = t[mask]
        y = y[mask]
        yerr = yerr[mask] if yerr is not None else None

    if no_background:
        # Fit without background
        A0 = y[0]
        lambd0 = np.log(2) / half_life if half_life else np.log(2) / 220
        if half_life:
            popt, pcov = curve_fit(
                lambda t, A: exponential_decay_no_bg(t, A, lambd0),
                t, y, sigma=yerr, p0=(A0,), absolute_sigma=True
            )
        else:
            popt, pcov = curve_fit(
                exponential_decay_no_bg, t, y, sigma=yerr, p0=(A0, lambd0), absolute_sigma=True
            )

        return popt, pcov
      
    else:
        bg = bg_dict['rates'] if bg_dict is not None else np.zeros_like(y)
        B0 = np.mean(bg)
        B_std = np.std(bg)
        bg_lower = B0 - B_std
        bg_upper = B0 + B_std
        bg_lower = max(bg_lower, 0)

        A0 = y[0] - B0
        lambd0 = np.log(2) / half_life if half_life else np.log(2) / 220

        if half_life:
            popt, pcov = curve_fit(
                lambda t, A, B: exponential_decay(t, A, lambd0, B),
                t, y, sigma=yerr, p0=(A0, B0), absolute_sigma=True, bounds=([0, bg_lower], [np.inf, bg_upper])
            )
        else:
            popt, pcov = curve_fit(
                exponential_decay, t, y, sigma=yerr, p0=(A0, lambd0, B0), absolute_sigma=True, bounds=([0, 0, bg_lower], [np.inf, np.inf, bg_upper])
            )
        return popt, pcov
    
def compute_integral_and_error(popt, pcov, integral_time_window, fixed_half_life=None, include_bg=True):
    t1, t2 = integral_time_window
    if fixed_half_life is not None:
        fixed_lambda = np.log(2) / fixed_half_life
    else:
        fixed_lambda = None

    if include_bg:
        if fixed_lambda is not None:
            A, B = popt
            lambd = fixed_lambda
            I = (A / lambd) * (np.exp(-lambd * t1) - np.exp(-lambd * t2)) + B * (t2 - t1)

            # Partial derivatives
            dI_dA = (1 / lambd) * (np.exp(-lambd * t1) - np.exp(-lambd * t2))
            dI_dB = t2 - t1

            var_I = (
                dI_dA**2 * pcov[0, 0] +
                dI_dB**2 * pcov[1, 1] +
                2 * dI_dA * dI_dB * pcov[0, 1]
            )

        else:
            A, lambd, B = popt
            I = (A / lambd) * (np.exp(-lambd * t1) - np.exp(-lambd * t2)) + B * (t2 - t1)

            dI_dA = (1 / lambd) * (np.exp(-lambd * t1) - np.exp(-lambd * t2))
            dI_dl = (-A / lambd**2) * (np.exp(-lambd * t1) - np.exp(-lambd * t2)) + \
                    (A / lambd) * (-t1 * np.exp(-lambd * t1) + t2 * np.exp(-lambd * t2))
            dI_dB = t2 - t1

            var_I = (
                dI_dA**2 * pcov[0, 0] +
                dI_dl**2 * pcov[1, 1] +
                dI_dB**2 * pcov[2, 2] +
                2 * dI_dA * dI_dl * pcov[0, 1] +
                2 * dI_dA * dI_dB * pcov[0, 2] +
                2 * dI_dl * dI_dB * pcov[1, 2]
            )
            # lambd and its error
            print(f"Lambda: {lambd:.4f} ± {np.sqrt(pcov[1, 1]):.4f}")
    else:
        if fixed_lambda is not None:
            A = popt[0]
            lambd = fixed_lambda
            I = (A / lambd) * (np.exp(-lambd * t1) - np.exp(-lambd * t2))
            dI_dA = (1 / lambd) * (np.exp(-lambd * t1) - np.exp(-lambd * t2))
            var_I = dI_dA**2 * pcov[0, 0]
        else:
            A, lambd, B = popt
            I = (A / lambd) * (np.exp(-lambd * t1) - np.exp(-lambd * t2))

            dI_dA = (1 / lambd) * (np.exp(-lambd * t1) - np.exp(-lambd * t2))
            dI_dl = (-A / lambd**2) * (np.exp(-lambd * t1) - np.exp(-lambd * t2)) + \
                    (A / lambd) * (-t1 * np.exp(-lambd * t1) + t2 * np.exp(-lambd * t2))

            var_I = (
                dI_dA**2 * pcov[0, 0] +
                dI_dl**2 * pcov[1, 1] +
                2 * dI_dA * dI_dl * pcov[0, 1]
            )

            print(f"Lambda: {lambd:.4f} ± {np.sqrt(pcov[1, 1]):.4f}")

    I_err = np.sqrt(var_I) if var_I >= 0 else np.nan
    return I, I_err

def generate_exponential_bins(half_life, t_min, t_max, N):
    """
    Generate exponentially spaced bins over [t_min, t_max] using the decay law.
    Each bin corresponds to equal area under the decay curve.
    """
    lmbd = np.log(2) / half_life
    T = t_max - t_min

    norm = 1 - np.exp(-lmbd * T)
    frac = np.linspace(0, 1, N + 1)
    return -np.log(1 - frac * norm) / lmbd + t_min

def correlate_binned_counts(B, G, C, B_back, G_back,
                             half_life, t_min, t_max, N,
                             exclude_bins=None):
    """
    Bins B, G, C (and their backgrounds) using exponential bins,
    subtracts background, excludes selected bins, and computes correlation matrix.

    Parameters:
        B, G, C: arrays of event timestamps [s]
        B_back, G_back: arrays of background timestamps [s]
        half_life: isotope half-life [s]
        t_min, t_max: range for binning [s]
        N: number of bins
        exclude_bins: list of bin indices to exclude (e.g., [0, 1])

    Returns:
        corr_matrix: 3x3 correlation matrix between [B_net, G_net, C_net] (after exclusions)
        binned_counts: dictionary with keys 'B', 'G', 'C' (each is full-length 1D array of size N)
    """
    # Generate bin edges
    bins = generate_exponential_bins(half_life, t_min, t_max, N)

    # Histogram signal and background
    hist_B = np.histogram(B, bins=bins)[0]
    hist_G = np.histogram(G, bins=bins)[0]
    hist_C = np.histogram(C, bins=bins)[0]

    hist_B_back = np.histogram(B_back, bins=bins)[0]
    hist_G_back = np.histogram(G_back, bins=bins)[0]

    # Background subtraction
    B_net = hist_B - hist_B_back
    G_net = hist_G - hist_G_back
    C_net = hist_C  # No background for coincidences

    # Exclude infected bins
    if exclude_bins:
        include_mask = np.ones(N, dtype=bool)
        include_mask[exclude_bins] = False

        B_corr = B_net[include_mask]
        G_corr = G_net[include_mask]
        C_corr = C_net[include_mask]
    else:
        B_corr, G_corr, C_corr = B_net, G_net, C_net

    # Compute correlation matrix
    data = np.vstack([B_corr, G_corr, C_corr])
    corr_matrix = np.corrcoef(data)

    binned_counts = {
        "Betas": B_net,
        "Gammas": G_net,
        "Coincidences": C_net,
    }

    return corr_matrix, binned_counts

def build_covariance_matrix(stds, correlation_matrix=None):
    cov = np.diag(stds ** 2)
    if correlation_matrix is not None:
        cov = np.outer(stds, stds) * correlation_matrix
    return cov

def filter_positive_samples(samples):
    return samples[(samples > 0).all(axis=1)]

def compute_efficiency(
    N_num, sigma_num,
    N_den, sigma_den,
    n_samples=1000000,
    correlation_matrix=None
):
    means = np.array([N_num, N_den])
    stds = np.array([sigma_num, sigma_den])
    cov = build_covariance_matrix(stds, correlation_matrix)
    samples = np.random.multivariate_normal(means, cov, size=n_samples)
    samples = filter_positive_samples(samples)
    eff_samples = (samples[:, 1]) / samples[:, 0]
    eff = 100 * np.mean(eff_samples)
    eff_err = 100 * np.std(eff_samples)
    return eff, eff_err

# For backward compatibility:
def compute_beta_efficiency(N_gamma, sigma_gamma, N_coinc, sigma_coinc, n_samples=1000000, correlation_matrix=None):
    return compute_efficiency(N_gamma, sigma_gamma, N_coinc, sigma_coinc, n_samples, correlation_matrix)

def compute_gamma_efficiency(N_beta, sigma_beta, N_coinc, sigma_coinc, n_samples=1000000, correlation_matrix=None):
    return compute_efficiency(N_beta, sigma_beta, N_coinc, sigma_coinc, n_samples, correlation_matrix)

def compute_number_of_decays(
    N_beta, sigma_beta,
    N_gamma, sigma_gamma,
    N_coinc, sigma_coinc,
    n_samples=1000000,
    correlation_matrix=None
):
    means = np.array([N_beta, N_gamma, N_coinc])
    stds = np.array([sigma_beta, sigma_gamma, sigma_coinc])
    cov = build_covariance_matrix(stds, correlation_matrix)

    # Sample from multivariate normal
    samples = np.random.multivariate_normal(means, cov, size=n_samples)

    samples = filter_positive_samples(samples)

    # Compute number of decays for each sample
    N_decay = (samples[:, 0] * samples[:, 1]) / samples[:, 2]

    # Get statistics
    N_decay_mean = np.mean(N_decay)
    N_decay_std = np.std(N_decay)

    return N_decay_mean, N_decay_std

def compute_accumulated_activity(
    N_beta, sigma_beta,
    N_gamma, sigma_gamma,
    N_coinc, sigma_coinc,
    half_life,
    integral_time_window,
    correlation_matrix=None,
    n_samples=1000000
):
    N_D, N_D_err = compute_number_of_decays(
        N_beta, sigma_beta,
        N_gamma, sigma_gamma,
        N_coinc, sigma_coinc,
        n_samples=n_samples,
        correlation_matrix=correlation_matrix
    )

    lambd = np.log(2) / half_life

    A_accum = (N_D * lambd) / (np.exp(-lambd * integral_time_window[0]) - np.exp(-lambd * integral_time_window[1]))
    A_accum_err = (N_D_err * lambd) / (np.exp(-lambd * integral_time_window[0]) - np.exp(-lambd * integral_time_window[1]))

    return A_accum, A_accum_err
