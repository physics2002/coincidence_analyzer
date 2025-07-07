import numpy as np
from scipy.optimize import curve_fit
from scipy.integrate import simps

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
            A_fit = popt[0]
            return (A_fit, lambd0), pcov
        else:
            popt, pcov = curve_fit(
                exponential_decay_no_bg, t, y, sigma=yerr, p0=(A0, lambd0), absolute_sigma=True
            )
            return popt, pcov
    else:
        bg = bg_dict['rates'] if bg_dict is not None else np.zeros_like(y)
        B0 = np.mean(bg)
        B_std = np.std(bg)
        bg_lower = B0 - 3 * B_std
        bg_upper = B0 + 3 * B_std
        bg_lower = max(bg_lower, 0)

        A0 = y[0] - B0
        lambd0 = np.log(2) / half_life if half_life else 0.001

        if half_life:
            popt, pcov = curve_fit(
                lambda t, A, B: exponential_decay(t, A, lambd0, B),
                t, y, sigma=yerr, p0=(A0, B0), absolute_sigma=True, bounds=([0, bg_lower], [np.inf, bg_upper])
            )
            A_fit, B_fit = popt
            return (A_fit, lambd0, B_fit), pcov
        else:
            popt, pcov = curve_fit(
                exponential_decay, t, y, sigma=yerr, p0=(A0, lambd0, B0), absolute_sigma=True, bounds=([0, 0, bg_lower], [np.inf, np.inf, bg_upper])
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