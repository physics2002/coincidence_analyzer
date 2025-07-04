import numpy as np
from scipy.optimize import curve_fit
from scipy.integrate import simps

def exponential_decay(t, A, lambd, B):
    return A * np.exp(-lambd * t) + B

def fit_decay_curve(signal_dict, bg_dict, half_life=None, time_window=None):
    """
    Fit count rate vs time data to exponential decay model.
    
    Parameters:
    - times: list or np.array of time bin centers (in seconds)
    - rates: list or np.array of count rates
    - errors: optional uncertainties on rates
    - half_life: optional half-life in seconds (fixes λ)
    - time_window: optional (t_min, t_max) for fitting window
    
    Returns:
    - popt: fitted parameters
    - pcov: covariance matrix
    """
    t = signal_dict['centers']
    y = signal_dict['rates']

    bg = bg_dict['rates']
    yerr = signal_dict['errors']

    # Apply time window filter
    if time_window:
        t_min, t_max = time_window
        mask = (t >= t_min) & (t <= t_max)
        t = t[mask]
        y = y[mask]
        bg = bg[mask] if len(bg) == len(y) else bg
        yerr = yerr[mask] if yerr is not None else None

    # Use mean background as initial guess for B0
    B0 = np.mean(bg)
    A0 = y[0] - B0
    lambd0 = np.log(2) / half_life if half_life else 0.001

    if half_life:
        popt, pcov = curve_fit(
            lambda t, A, B: exponential_decay(t, A, lambd0, B),
            t, y, sigma=yerr, p0=(A0, B0), absolute_sigma=True
        )
        A_fit, B_fit = popt
        return (A_fit, lambd0, B_fit), pcov
    else:
        popt, pcov = curve_fit(
            exponential_decay, t, y, sigma=yerr, p0=(A0, lambd0, B0), absolute_sigma=True
        )
        return popt, pcov
    
def integrate_counts(times, rates, time_window=None):
    """
    Integrate count rates numerically over an optional time window.
    """
    t = np.array(times)
    y = np.array(rates)

    if time_window:
        t_min, t_max = time_window
        mask = (t >= t_min) & (t <= t_max)
        t = t[mask]
        y = y[mask]

    return simps(y, t)