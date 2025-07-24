import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from coincidence_analysis.detector import DetectorData
from coincidence_analysis.coincidence import CoincidenceAnalyzer
from coincidence_analysis.plotting import (
    select_energy_range, select_time_window, plot_count_rates, select_fit_time_window
)
from coincidence_analysis.utils import compute_count_rate, get_times_in_energy_range
from coincidence_analysis.analysis import (
    exponential_decay, exponential_decay_no_bg, fit_decay_curve, 
    compute_integral_and_error, compute_accumulated_activity, compute_beta_efficiency, compute_gamma_efficiency
)

def save_background(backgroud_path):
    background = DetectorData(backgroud_path)
    background_data = background.get_dataframe_time_shifted(600)
    background_data.to_csv("background.txt", sep=";", index=False)

def load_data(signal_path, background_path):
    signal = DetectorData(signal_path)
    background = DetectorData(background_path)
    return signal.get_dataframe(), background.get_dataframe()

def analyze_coincidences(df_signal, df_background):
    analyzer = CoincidenceAnalyzer(df_signal)

    # Δt histogram and window selection
    delta_ts = analyzer.coincidence_search(dt_window_ns=(-200, 200), return_delta_t=True)
    dt_window = select_time_window(delta_ts, time_range=(-200, 200))#preselected_range=(-8.27, 177.04)

    coincidences_all = analyzer.coincidence_search(dt_window_ns=dt_window)

    # Energy selection
    range0 = select_energy_range(df_signal, channel=0, bins=2000, energy_range=(0, 50000), coincidences=coincidences_all, 
                                 df_background=df_background, preselected_range=(9097, 11182))
    range1 = select_energy_range(df_signal, channel=1, bins=1000, energy_range=(0, 50000), 
                                 df_background=df_background, preselected_range=(0, 50000))

    # Coincidence analysis
    coincidences_in_range = analyzer.coincidence_search(
        dt_window_ns=dt_window,
        energy_range_a=range0,
        energy_range_b=range1
    )

    return (coincidences_in_range, range0, range1)

def compute_and_plot_rates(df_signal, df_bg, coincidences_in_range, range0, range1, bin_width_s=3.0):
    ch0_times = get_times_in_energy_range(df_signal, channel=0, energy_range=range0)
    ch1_times = get_times_in_energy_range(df_signal, channel=1, energy_range=range1)
    coin_times = [min(pair[0].time_s, pair[1].time_s) for pair in coincidences_in_range]

    bg_ch0_times = get_times_in_energy_range(df_bg, channel=0, energy_range=range0)
    bg_ch1_times = get_times_in_energy_range(df_bg, channel=1, energy_range=range1)

    signal_ch0 = compute_count_rate(ch0_times, bin_width_s=bin_width_s)
    signal_ch1 = compute_count_rate(ch1_times, bin_width_s=bin_width_s)
    signal_coinc = compute_count_rate(coin_times, bin_width_s=bin_width_s*2)

    background_ch0 = compute_count_rate(bg_ch0_times, bin_width_s=bin_width_s)
    background_ch1 = compute_count_rate(bg_ch1_times, bin_width_s=bin_width_s)

    # Fit decay curves
    time_window_0 = select_fit_time_window(signal_ch0)
    time_window_1 = select_fit_time_window(signal_ch1)
    time_window_coin = select_fit_time_window(signal_coinc)

    half_life = 134.7  # in seconds

    popt_ch0, pcov_ch0 = fit_decay_curve(signal_ch0, background_ch0, half_life=half_life, time_window=time_window_0)
    popt_ch1, pcov_ch1 = fit_decay_curve(signal_ch1, background_ch1, half_life=half_life, time_window=time_window_1)
    popt_coin, pcov_coin = fit_decay_curve(signal_coinc, half_life=half_life, time_window=time_window_coin, no_background=True)

    #Compute integrals
    integral_time_window = (0, 600)
    N_beta, N_beta_err = compute_integral_and_error(popt_ch0, pcov_ch0, integral_time_window, fixed_half_life=half_life, include_bg=False)
    N_gamma, N_gamma_err = compute_integral_and_error(popt_ch1, pcov_ch1, integral_time_window, fixed_half_life=half_life, include_bg=False)
    N_coinc, N_coinc_err = compute_integral_and_error(popt_coin, pcov_coin, integral_time_window, fixed_half_life=half_life, include_bg=False)

    plot_count_rates(
        signal_ch0=signal_ch0,
        signal_ch1=signal_ch1,
        signal_coinc=signal_coinc,
        background_ch0=background_ch0,
        background_ch1=background_ch1,
        fit_func_ch0=exponential_decay, popt_ch0=popt_ch0,
        fit_func_ch1=exponential_decay, popt_ch1=popt_ch1,
        fit_func_coinc=exponential_decay_no_bg, popt_coinc=popt_coin,
        half_life=half_life,
        time_window=integral_time_window
    )

    print(f"Integrals over time window {integral_time_window[0]}s to {integral_time_window[1]}s:")
    print(f"Beta decay integral:  {N_beta:.2f} ± {N_beta_err:.2f}")
    print(f"Gamma decay integral: {N_gamma:.2f} ± {N_gamma_err:.2f}")
    print(f"Coincidence integral: {N_coinc:.2f} ± {N_coinc_err:.2f}")

    e_b, e_b_err = compute_beta_efficiency(N_gamma=N_gamma, sigma_gamma=N_gamma_err,
                                            N_coinc=N_coinc, sigma_coinc=N_coinc_err,
                                            correlation_matrix=None)
    
    e_g, e_g_err = compute_gamma_efficiency(N_beta=N_beta, sigma_beta=N_beta_err,
                                            N_coinc=N_coinc, sigma_coinc=N_coinc_err,
                                            correlation_matrix=None)
    print(f"Beta efficiency: {e_b:.2f} ± {e_b_err:.2f}")
    print(f"Gamma efficiency: {e_g:.2f} ± {e_g_err:.2f}")

    A_accum, A_accum_err = compute_accumulated_activity(
        N_beta=N_beta, sigma_beta=N_beta_err, 
        N_gamma=N_gamma, sigma_gamma=N_gamma_err, 
        N_coinc=N_coinc, sigma_coinc=N_coinc_err,
        half_life=half_life, integral_time_window=integral_time_window,
        correlation_matrix=None)
    print(f"Accumulated activity: ({A_accum:.2f} ± {A_accum_err:.2f}) Bq")

def main():
    signal_path = r'Z:\Studenten\Bakhodirov\coincidence_analyzer-1\coincidence_analysis\Detector_data\AlSurf280um_listmode.txt'
    background_path = r'Z:\Studenten\Bakhodirov\coincidence_analyzer-1\coincidence_analysis\Detector_data\background_listmode.txt'

    df_signal, df_bg = load_data(signal_path, background_path)
    (coincidences_in_range, range0, range1) = analyze_coincidences(df_signal, df_bg)
    compute_and_plot_rates(df_signal, df_bg, coincidences_in_range, range0, range1)

if __name__ == "__main__":
    main()