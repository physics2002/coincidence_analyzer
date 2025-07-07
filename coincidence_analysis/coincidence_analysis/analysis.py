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