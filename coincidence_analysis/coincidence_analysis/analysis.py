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
        lambd0 = np.log(2) / half_life if half_life else 0.001
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
        lambd0 = np.log(2) / half_life if half_life else 0.001

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
    else:
        if fixed_lambda is not None:
            A = popt[0]
            lambd = fixed_lambda
            I = (A / lambd) * (np.exp(-lambd * t1) - np.exp(-lambd * t2))
            dI_dA = (1 / lambd) * (np.exp(-lambd * t1) - np.exp(-lambd * t2))
            var_I = dI_dA**2 * pcov[0, 0]
        else:
            A, lambd = popt
            I = (A / lambd) * (np.exp(-lambd * t1) - np.exp(-lambd * t2))

            dI_dA = (1 / lambd) * (np.exp(-lambd * t1) - np.exp(-lambd * t2))
            dI_dl = (-A / lambd**2) * (np.exp(-lambd * t1) - np.exp(-lambd * t2)) + \
                    (A / lambd) * (-t1 * np.exp(-lambd * t1) + t2 * np.exp(-lambd * t2))

            var_I = (
                dI_dA**2 * pcov[0, 0] +
                dI_dl**2 * pcov[1, 1] +
                2 * dI_dA * dI_dl * pcov[0, 1]
            )

    I_err = np.sqrt(var_I) if var_I >= 0 else np.nan
    return I, I_err

def compute_beta_efficiency(
    N_gamma, sigma_gamma,
    N_coinc, sigma_coinc,
    n_samples=1000000,
    correlation_matrix=None
):
    means = np.array([N_gamma, N_coinc])
    stds = np.array([sigma_gamma, sigma_coinc])
    cov = np.diag(stds ** 2)

    if correlation_matrix is not None:
        cov = np.outer(stds, stds) * correlation_matrix
    
    samples = np.random.multivariate_normal(means, cov, size=n_samples)
    samples = samples[(samples[:, 0] > 0) & (samples[:, 1] > 0)]

    e_b_samples = (samples[:, 1]) / samples[:, 0]

    e_b = 100*np.mean(e_b_samples)
    e_b_err = 100*np.std(e_b_samples)

    return e_b, e_b_err

def compute_gamma_efficiency(
    N_beta, sigma_beta,
    N_coinc, sigma_coinc,
    n_samples=1000000,
    correlation_matrix=None
):
    means = np.array([N_beta, N_coinc])
    stds = np.array([sigma_beta, sigma_coinc])
    cov = np.diag(stds ** 2)

    if correlation_matrix is not None:
        cov = np.outer(stds, stds) * correlation_matrix
    
    samples = np.random.multivariate_normal(means, cov, size=n_samples)
    samples = samples[(samples[:, 0] > 0) & (samples[:, 1] > 0)]

    e_g_samples = (samples[:, 1]) / samples[:, 0]

    e_g = 100*np.mean(e_g_samples)
    e_g_err = 100*np.std(e_g_samples)

    return e_g, e_g_err

def compute_number_of_decays(
    N_beta, sigma_beta,
    N_gamma, sigma_gamma,
    N_coinc, sigma_coinc,
    n_samples=1000000,
    correlation_matrix=None
):
    means = np.array([N_beta, N_gamma, N_coinc])
    stds = np.array([sigma_beta, sigma_gamma, sigma_coinc])
    cov = np.diag(stds ** 2)

    if correlation_matrix is not None:
        # Convert correlation matrix to covariance matrix
        cov = np.outer(stds, stds) * correlation_matrix

    # Sample from multivariate normal
    samples = np.random.multivariate_normal(means, cov, size=n_samples)

    # Avoid division by zero or negative counts
    samples = samples[(samples[:, 0] > 0) & (samples[:, 1] > 0) & (samples[:, 2] > 0)]

    # Compute activity for each sample
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
