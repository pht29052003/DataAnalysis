import sys
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt, butter, filtfilt, savgol_filter
from scipy.stats import skew, norm
from scipy.optimize import minimize
import pywt
import time
from sklearn.mixture import GaussianMixture

# ==========================================
# 1. NOISE FILTERING & MATHEMATICAL FUNCTIONS
# ==========================================
def moving_average(data, window_size):
    return pd.Series(data).rolling(window=window_size, min_periods=1, center=True).mean().values

def butter_lowpass_filter(data, cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

def wavelet_denoise(data, wavelet='db4', level=1):
    coeff = pywt.wavedec(data, wavelet, mode="per")
    sigma = (1/0.6745) * np.median(np.abs(coeff[-level] - np.median(coeff[-level])))
    if sigma == 0: sigma = 1e-4 
    uthresh = sigma * np.sqrt(2 * np.log(len(data)))
    coeff[1:] = [pywt.threshold(i, value=uthresh, mode='soft') for i in coeff[1:]]
    reconstructed_signal = pywt.waverec(coeff, wavelet, mode='per')
    return reconstructed_signal[:len(data)]

def kalman_filter(data, Q, R):
    n = len(data)
    xhat = np.zeros(n)
    P = np.zeros(n)
    xhatminus = np.zeros(n)
    Pminus = np.zeros(n)
    K = np.zeros(n)
    xhat[0] = data[0]
    P[0] = 1.0
    for k in range(1, n):
        xhatminus[k] = xhat[k-1]
        Pminus[k] = P[k-1] + Q
        K[k] = Pminus[k] / (Pminus[k] + R)
        xhat[k] = xhatminus[k] + K[k] * (data[k] - xhatminus[k])
        P[k] = (1 - K[k]) * Pminus[k]
    return xhat

# ==========================================
# 2. ALLAN DEVIATION ALGORITHM
# ==========================================
def calculate_allan_deviation(data, fs):
    N = len(data)
    t0 = 1.0 / fs
    theta = np.concatenate(([0], np.cumsum(data) * t0))
    max_m = N // 3 
    m_arr = np.unique(np.logspace(0, np.log10(max_m), 100).astype(int))
    m_arr = m_arr[m_arr > 0]
    taus = []
    ad = []
    for m in m_arr:
        tau = m * t0
        delta = theta[2*m:] - 2 * theta[m:-m] + theta[:-2*m]
        avar = np.sum(delta**2) / (2 * (tau**2) * (N - 2*m))
        taus.append(tau)
        ad.append(np.sqrt(avar))
    return np.array(taus), np.array(ad)

# ==========================================
# 3. EVALUATION FUNCTION (3 GAUSSIAN)
# ==========================================
def calculate_metrics(original, filtered):
    corr = np.corrcoef(original, filtered)[0, 1]
    tv_original = np.sum(np.abs(np.diff(original)))
    tv_filtered = np.sum(np.abs(np.diff(filtered)))
    tv_ratio = tv_filtered / (tv_original + 1e-10) 
    residual = original - filtered
    max_abs_error = np.max(np.abs(residual))
    try:
        res_skew = abs(skew(residual))
    except:
        res_skew = 0.0
    energy_ratio = np.sum(filtered**2) / (np.sum(original**2) + 1e-10)
    score = (corr**3 * 100) - (tv_ratio * 50)
    return corr, tv_ratio, max_abs_error, res_skew, energy_ratio, score

# ------------------------------------------
# 4. HYBRID MODEL SOLVER (2 GAUSSIAN + 1 RECTANGULAR)
# ------------------------------------------
def fit_2g_1u_model(data):
    if len(data) > 5000:
        np.random.seed(42)
        opt_data = np.random.choice(data, size=5000, replace=False)
    else:
        opt_data = data

    a, b = np.min(data), np.max(data)
    val_range = b - a if b > a else 1e-6
    u_pdf = 1.0 / val_range 

    gmm = GaussianMixture(n_components=2, random_state=42, n_init=1)
    gmm.fit(opt_data.reshape(-1, 1))
    mu1_init, mu2_init = gmm.means_.flatten()
    
    cov = gmm.covariances_
    if cov.ndim == 3: 
        sig1_init, sig2_init = np.sqrt(cov[0,0,0]), np.sqrt(cov[1,0,0])
    else: 
        sig1_init = sig2_init = np.sqrt(cov.flatten()[0])
    w1_init, w2_init = gmm.weights_

    def objective(params):
        mu1, mu2, sig1, sig2, wu, wg1 = params
        if wu + wg1 >= 1.0 or wu < 0 or wg1 < 0: return 1e10 
        wg2 = 1.0 - wu - wg1

        p_g1 = norm.pdf(opt_data, loc=mu1, scale=sig1)
        p_g2 = norm.pdf(opt_data, loc=mu2, scale=sig2)

        mix_pdf = wu * u_pdf + wg1 * p_g1 + wg2 * p_g2
        mix_pdf = np.clip(mix_pdf, 1e-300, None)
        return -np.sum(np.log(mix_pdf))

    bounds = [
        (a, b), (a, b),
        (val_range*1e-4, val_range), (val_range*1e-4, val_range),
        (0.001, 0.5), (0.001, 0.99) 
    ]
    init_params = [mu1_init, mu2_init, sig1_init, sig2_init, 0.05, w1_init*0.9]

    try:
        res = minimize(objective, init_params, bounds=bounds, method='L-BFGS-B')
        mu1, mu2, sig1, sig2, wu, wg1 = res.x
        wg2 = 1.0 - wu - wg1
    except:
        mu1, mu2, sig1, sig2, wu, wg1, wg2 = mu1_init, mu2_init, sig1_init, sig2_init, 0.0, w1_init, w2_init

    return mu1, mu2, sig1, sig2, wu, wg1, wg2, a, b

# ==========================================
# 4. MAIN STREAMLIT INTERFACE
# ==========================================
def main():
    st.set_page_config(layout="wide", page_title="Data Analysis Dashboard")
    st.title("Data Analysis")

    st.sidebar.header("1. Open Data File")
    uploaded_file = st.sidebar.file_uploader("Select CSV/Excel file", type=['csv', 'xlsx'])

    if uploaded_file is not None:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file, header=None)
        else:
            df = pd.read_excel(uploaded_file, header=None)
            
        head_text = df.head(5).to_string().lower()
        if any(keyword in head_text for keyword in ['current', 'ma', 'amp']):
            default_index = 1 
        else:
            default_index = 0
            
        st.sidebar.header("2. Data Formatting")
        daq_type = st.sidebar.radio("Detected raw signal type:", ("Voltage", "Current"), index=default_index)
        
        raw_col_name = 'Voltage' if daq_type == "Voltage" else 'Current'
        df.columns = ['Time', raw_col_name, 'Pressure'] + list(df.columns[3:])
        
        st.sidebar.header("3. Select Signal")
        signal_choice = st.sidebar.radio("Select signal to denoise:", (raw_col_name, 'Pressure'))
        
        df['Time'] = pd.to_numeric(df['Time'], errors='coerce')
        df[signal_choice] = pd.to_numeric(df[signal_choice], errors='coerce')
        df_clean = df.dropna(subset=['Time', signal_choice])
        
        try:
            mean_dt = np.mean(np.diff(df_clean['Time'].values[:1000])) 
            auto_fs = int(round(1.0 / mean_dt)) if mean_dt > 0 else 1000
        except:
            auto_fs = 1000
            
        st.sidebar.success(f"Sampling frequency: **{auto_fs:,} Hz**")
        
        st.sidebar.header("4. Auxiliary Filter Parameters")
        fs = st.sidebar.number_input("Confirm Sampling Frequency (Hz)", value=auto_fs)
        window_ma = st.sidebar.slider("Window Size (Standard MA & Median)", 3, 101, 15, step=2)
        cutoff = st.sidebar.slider("Cutoff Frequency (Butterworth) - Hz", 1.0, fs/2 - 1.0, 50.0)
        wavelet_type = st.sidebar.selectbox("Wavelet Type", ['db4', 'sym4', 'coif3'])
        Q_kalman = st.sidebar.number_input("Kalman Noise (Q)", value=1e-5, format="%e")
        R_kalman = st.sidebar.number_input("Kalman Noise (R)", value=0.01, format="%f")
        window_sg = st.sidebar.slider("Window Size (Savitzky-Golay)", 5, 101, 21, step=2)
        poly_sg = st.sidebar.slider("Polynomial Order (Savitzky-Golay)", 1, 5, 3)

        time_arr = df_clean['Time'].values.copy()
        raw_signal = df_clean[signal_choice].values.copy()
        total_points = len(raw_signal)

        st.info(f"Loading **{total_points:,}** data points. Running Allan Deviation to find optimal Window Size...")

        with st.spinner('Processing potentially millions of data points. Feel free to grab a coffee while you wait...'):
            
            taus_raw, ad_raw = calculate_allan_deviation(raw_signal, fs)
            min_idx = np.argmin(ad_raw)
            opt_tau = taus_raw[min_idx]       
            opt_ad = ad_raw[min_idx]          
            opt_window = int(opt_tau * fs)
            if opt_window % 2 == 0: opt_window += 1 
            if opt_window < 3: opt_window = 3
            
            results = {}
            exec_times = {}
            
            start = time.time()
            med_trimmed = medfilt(raw_signal, kernel_size=5)
            results['Hybrid Optimal'] = moving_average(med_trimmed, window_size=opt_window)
            exec_times['Hybrid Optimal'] = time.time() - start

            start = time.time()
            results['Median'] = medfilt(raw_signal, kernel_size=window_ma)
            exec_times['Median'] = time.time() - start
            
            start = time.time()
            results['Moving Average'] = moving_average(raw_signal, window_size=window_ma)
            exec_times['Moving Average'] = time.time() - start
            
            start = time.time()
            results['Butterworth'] = butter_lowpass_filter(raw_signal, cutoff, fs)
            exec_times['Butterworth'] = time.time() - start
            
            start = time.time()
            results['Wavelet'] = wavelet_denoise(raw_signal, wavelet=wavelet_type)
            exec_times['Wavelet'] = time.time() - start
            
            start = time.time()
            results['Kalman'] = kalman_filter(raw_signal, Q_kalman, R_kalman)
            exec_times['Kalman'] = time.time() - start
            
            try:
                start = time.time()
                results['Savitzky-Golay'] = savgol_filter(raw_signal, window_length=window_sg, polyorder=poly_sg)
                exec_times['Savitzky-Golay'] = time.time() - start
            except:
                results['Savitzky-Golay'] = raw_signal 
                exec_times['Savitzky-Golay'] = 0.0

            metrics_data = []
            methods = ['Hybrid Optimal', 'Median', 'Moving Average', 'Butterworth', 'Wavelet', 'Kalman', 'Savitzky-Golay']
            for method in methods:
                corr, tv_ratio, max_abs_error, res_skew, energy_ratio, score = calculate_metrics(raw_signal, results[method])
                metrics_data.append([method, round(corr, 4), round(tv_ratio, 4), round(max_abs_error, 4),
                                     round(res_skew, 4), round(energy_ratio, 4), round(exec_times[method], 4), round(score, 2)])

            df_metrics = pd.DataFrame(metrics_data, columns=['Method', 'Tracking (~1.0)', 'Fluctuation (~0.0)', 'Max Spike Removal',
                                                             'Distortion (~0.0)', 'Energy Conservation', 'CPU Speed', 'Total Score'])
            df_metrics = df_metrics.sort_values(by='Total Score', ascending=False).reset_index(drop=True)
            df_metrics.insert(0, 'No.', range(1, len(df_metrics) + 1))

            st.subheader("Noise Filtering Efficiency Leaderboard")
            st.dataframe(df_metrics.style.highlight_max(subset=['Total Score'], color='lightgreen'), use_container_width=True, hide_index=True)
                         
            # ==========================================
            # 4.1: TIME DOMAIN
            # ==========================================
            st.markdown("---")
            st.subheader("1. Time Domain Graph (100% Full Width)")
            
            fig, axes = plt.subplots(4, 2, figsize=(18, 16), sharex=True)
            axes = axes.flatten()
            for i, method in enumerate(methods):
                axes[i].plot(time_arr, raw_signal, color='lightgray', label='Raw', alpha=0.7)
                col = 'red' if method == 'Hybrid Optimal' else 'blue'
                lw = 1.8 if method == 'Hybrid Optimal' else 1.2
                axes[i].plot(time_arr, results[method], color=col, label='Filtered', linewidth=lw)
                axes[i].set_title(f"{method}" if method == 'Hybrid Optimal' else f"{method} Filter", fontweight='bold' if method == 'Hybrid Optimal' else 'normal')
                axes[i].grid(True, linestyle='--', alpha=0.6)
                axes[i].set_xlim([time_arr.min(), time_arr.max()])
                if i >= 6: axes[i].set_xlabel("Time (s)")
            
            if len(methods) % 2 != 0: axes[-1].axis('off')
            plt.tight_layout()
            st.pyplot(fig)

            # ==========================================
            # 4.2: ALLAN DEVIATION
            # ==========================================
            st.markdown("---")
            st.subheader("2. Allan Deviation Analysis")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric(label="⏱Optimal Sample Time (τ)", value=f"{opt_tau:.4f} s")
            col2.metric(label="Effective Sample Rate", value=f"{1.0/opt_tau:.2f} Hz")
            col3.metric(label="Optimal Window Size", value=f"{opt_window} points")
            col4.metric(label="Bias Instability (Floor)", value=f"{opt_ad:.2e}")

            ad_plot_placeholder = st.empty()
            st.markdown("<br>**Allan Graph Display Options (Tick to show/hide):**", unsafe_allow_html=True)
            chk_cols = st.columns(4)
            show_raw = chk_cols[0].checkbox(f"Raw {signal_choice}", value=True)
            show_opt = chk_cols[1].checkbox("Optimal Point", value=True)
            
            show_methods = {}
            for i, method in enumerate(methods):
                col_idx = (i + 2) % 4 
                show_methods[method] = chk_cols[col_idx].checkbox(method, value=True)

            fig_ad, ax_ad = plt.subplots(figsize=(12, 6))
            if show_raw: ax_ad.loglog(taus_raw, ad_raw, linestyle='--', color='gray', linewidth=2.5, label=f'Raw {signal_choice}')
            if show_opt: ax_ad.scatter(opt_tau, opt_ad, color='red', s=150, zorder=5, label='Optimal Point')
            
            colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan', 'magenta']
            for i, method in enumerate(methods):
                if show_methods[method]:
                    taus_filt, ad_filt = calculate_allan_deviation(results[method], fs)
                    lw = 2.5 if method == 'Hybrid Optimal' else 1.5
                    ax_ad.loglog(taus_filt, ad_filt, color=colors[i], linewidth=lw, label=f'{method}', alpha=0.9)
            
            ax_ad.set_title('Allan Deviation', fontsize=14, fontweight='bold')
            ax_ad.set_xlabel('Averaging Time ($\\tau$) [s]', fontsize=12)
            ax_ad.set_ylabel('Allan Deviation ($\\sigma_A$)', fontsize=12)
            ax_ad.grid(True, which="both", ls="--", alpha=0.5)
            if show_raw or show_opt or any(show_methods.values()): ax_ad.legend(loc="lower left")
            ad_plot_placeholder.pyplot(fig_ad)

            # ==========================================
            # 4.3: 3 Gaussian Distribution
            # ==========================================
            st.markdown("---")
            st.subheader("3. 3-Component Gaussian Model Analysis")
            
            gmm_col1, gmm_col2 = st.columns(2)
            cov_val = gmm_col1.selectbox("Covariance Configuration:", ['full', 'tied', 'diag', 'spherical'], index=0)
            b_val = gmm_col2.slider("Histogram Detail (Bins):", 10, 200, 100)

            gmm_targets = ['Raw'] + methods
            gmm_progress = st.progress(0)
            gmm_status = st.empty()

            for idx, gmm_target in enumerate(gmm_targets):
                gmm_status.text(f"Analyzing GMM for signal: {gmm_target} ({idx+1}/{len(gmm_targets)})...")
                y_gmm = raw_signal if gmm_target == 'Raw' else results[gmm_target]
                
                trim_points = int(2 * fs)
                if len(y_gmm) > trim_points * 2:
                    y_gmm_trimmed = y_gmm[trim_points:]
                    x_gmm_trimmed = time_arr[trim_points:]
                else:
                    y_gmm_trimmed = y_gmm
                    x_gmm_trimmed = time_arr

                if len(y_gmm_trimmed) >= 3:
                    Y = y_gmm_trimmed.reshape(-1, 1)
                    gmm = GaussianMixture(n_components=3, random_state=42, n_init=10, covariance_type=cov_val)
                    gmm.fit(Y)
                    
                    weights = gmm.weights_
                    means = gmm.means_.flatten()
                    
                    if cov_val == 'full':
                        stds = np.sqrt(gmm.covariances_.reshape(3))
                    elif cov_val == 'tied':
                        shared_std = np.sqrt(gmm.covariances_[0, 0])
                        stds = [shared_std, shared_std, shared_std]
                    elif cov_val == 'diag':
                        stds = np.sqrt(gmm.covariances_.flatten())
                    elif cov_val == 'spherical':
                        stds = np.sqrt(gmm.covariances_)

                    fig_gmm, (ax1_gmm, ax2_gmm, ax3_gmm) = plt.subplots(1, 3, figsize=(15, 5), gridspec_kw={'width_ratios': [1, 1, 0.25]})

                    ax1_gmm.hist(y_gmm_trimmed, bins=b_val, color='skyblue', edgecolor='black', alpha=0.6, density=True)
                    x_pdf = np.linspace(Y.min(), Y.max(), 1000)
                    
                    pdf1 = weights[0] * norm.pdf(x_pdf, means[0], stds[0])
                    pdf2 = weights[1] * norm.pdf(x_pdf, means[1], stds[1])
                    pdf3 = weights[2] * norm.pdf(x_pdf, means[2], stds[2])
                    pdf_combined = pdf1 + pdf2 + pdf3
                    
                    ax1_gmm.plot(x_pdf, pdf1, color='green', linestyle='--', linewidth=1.5, label=f'GMM 1 (μ={means[0]:.4f})')
                    ax1_gmm.plot(x_pdf, pdf2, color='purple', linestyle='--', linewidth=1.5, label=f'GMM 2 (μ={means[1]:.4f})')
                    ax1_gmm.plot(x_pdf, pdf3, color='brown', linestyle='--', linewidth=1.5, label=f'GMM 3 (μ={means[2]:.4f})')
                    ax1_gmm.plot(x_pdf, pdf_combined, color='red', linestyle='-', linewidth=2, label='Combined GMM')
                    
                    ax1_gmm.set_title(f'Data: {gmm_target}')
                    ax1_gmm.set_xlabel(signal_choice)
                    ax1_gmm.set_ylabel('Probability Density')
                    ax1_gmm.legend(loc='upper right', fontsize=8)
                    ax1_gmm.grid(True, linestyle=':', alpha=0.6)

                    ax2_gmm.scatter(x_gmm_trimmed, y_gmm_trimmed, alpha=0.5, c='orange', s=15)
                    ax2_gmm.axhline(means[0], color='green', linestyle='--', linewidth=1.5, alpha=0.9)
                    ax2_gmm.axhline(means[1], color='purple', linestyle='--', linewidth=1.5, alpha=0.9)
                    ax2_gmm.axhline(means[2], color='brown', linestyle='--', linewidth=1.5, alpha=0.9)
                    ax2_gmm.set_title('Scatter Plot (Skipping 2s transient)')
                    ax2_gmm.set_xlim([x_gmm_trimmed.min(), x_gmm_trimmed.max()])
                    ax2_gmm.grid(True, linestyle=':', alpha=0.6)

                    medianprops = dict(linestyle='-', linewidth=2, color='firebrick')
                    flierprops = dict(marker='o', markersize=3, alpha=0.3)
                    ax3_gmm.boxplot(y_gmm_trimmed, vert=True, patch_artist=True,
                                boxprops=dict(facecolor='lightgreen', color='black'),
                                medianprops=medianprops,
                                showmeans=True, meanline=True, flierprops=flierprops)
                    ax3_gmm.set_yticklabels([])
                    ax3_gmm.set_xticklabels([])
                    ax3_gmm.set_title('Boxplot')
                    ax3_gmm.grid(True, linestyle=':', alpha=0.6, axis='y')

                    plt.tight_layout()
                    st.pyplot(fig_gmm)
                    plt.close(fig_gmm)

                gmm_progress.progress((idx + 1) / len(gmm_targets))
            gmm_status.success("Completed 3-Component Gaussian Distribution Analysis!")

            # ==========================================
            # 4.4: 2 GAUSSIAN + 1 RECTANGULAR
            # ==========================================
            st.markdown("---")
            st.subheader("4. 2 Gaussian + 1 Rectangular Model Analysis")

            hyb_progress = st.progress(0)
            hyb_status = st.empty()

            for idx, target in enumerate(gmm_targets):
                hyb_status.text(f"Solving optimization equation (MLE) for: {target} ({idx+1}/{len(gmm_targets)})...")
                
                y_hyb = raw_signal if target == 'Raw' else results[target]
                
                trim_points = int(2 * fs)
                if len(y_hyb) > trim_points * 2:
                    y_hyb_trimmed = y_hyb[trim_points:]
                    x_hyb_trimmed = time_arr[trim_points:]
                else:
                    y_hyb_trimmed = y_hyb
                    x_hyb_trimmed = time_arr

                mu1, mu2, sig1, sig2, wu, wg1, wg2, data_min, data_max = fit_2g_1u_model(y_hyb_trimmed)

                fig_hybrid, (ax1_hyb, ax2_hyb, ax3_hyb) = plt.subplots(1, 3, figsize=(15, 5), gridspec_kw={'width_ratios': [1, 1, 0.25]})
                
                ax1_hyb.hist(y_hyb_trimmed, bins=b_val, color='lightgray', edgecolor='black', alpha=0.5, density=True)
                
                x_pdf_hyb = np.linspace(data_min, data_max, 1000)
                
                pdf_u = wu * (np.ones_like(x_pdf_hyb) / (data_max - data_min if data_max > data_min else 1e-6))
                pdf_g1 = wg1 * norm.pdf(x_pdf_hyb, mu1, sig1)
                pdf_g2 = wg2 * norm.pdf(x_pdf_hyb, mu2, sig2)
                pdf_mix_hyb = pdf_u + pdf_g1 + pdf_g2

                ax1_hyb.plot(x_pdf_hyb, pdf_u, color='black', linestyle='-.', linewidth=1.5, label=f'Uniform ({wu*100:.1f}%)')
                ax1_hyb.plot(x_pdf_hyb, pdf_g1, color='green', linestyle='--', linewidth=1.5, label=f'GMM 1 (μ={mu1:.4f}, {wg1*100:.1f}%)')
                ax1_hyb.plot(x_pdf_hyb, pdf_g2, color='purple', linestyle='--', linewidth=1.5, label=f'GMM 2 (μ={mu2:.4f}, {wg2*100:.1f}%)')
                ax1_hyb.plot(x_pdf_hyb, pdf_mix_hyb, color='red', linestyle='-', linewidth=2.5, label='Hybrid PDF')

                ax1_hyb.set_title(f'Hybrid Model: {target}')
                ax1_hyb.set_xlabel(signal_choice)
                ax1_hyb.set_ylabel('Probability Density')
                ax1_hyb.legend(loc='upper right', fontsize=8)
                ax1_hyb.grid(True, linestyle=':', alpha=0.6)

                ax2_hyb.scatter(x_hyb_trimmed, y_hyb_trimmed, alpha=0.5, c='orange', s=15)
                ax2_hyb.axhline(mu1, color='green', linestyle='--', linewidth=1.5, alpha=0.9)
                ax2_hyb.axhline(mu2, color='purple', linestyle='--', linewidth=1.5, alpha=0.9)
                ax2_hyb.set_title('Scatter Plot (Skipping 2s transient)')
                ax2_hyb.set_xlabel('Time (s)')
                ax2_hyb.set_ylabel(signal_choice)
                ax2_hyb.set_xlim([x_hyb_trimmed.min(), x_hyb_trimmed.max()])
                ax2_hyb.grid(True, linestyle=':', alpha=0.6)

                medianprops = dict(linestyle='-', linewidth=2, color='firebrick')
                flierprops = dict(marker='o', markersize=3, alpha=0.3)
                ax3_hyb.boxplot(y_hyb_trimmed, vert=True, patch_artist=True,
                            boxprops=dict(facecolor='lightgray', color='black'),
                            medianprops=medianprops,
                            showmeans=True, meanline=True, flierprops=flierprops)
                ax3_hyb.set_yticklabels([])
                ax3_hyb.set_xticklabels([])
                ax3_hyb.set_title('Boxplot')
                ax3_hyb.grid(True, linestyle=':', alpha=0.6, axis='y')

                plt.tight_layout()
                st.pyplot(fig_hybrid)
                plt.close(fig_hybrid)

                hyb_progress.progress((idx + 1) / len(gmm_targets))

            hyb_status.success("Mathematical model solved for all filtering methods!")

            # ==========================================
            # 5. STORAGE / EXPORT
            # ==========================================
            st.markdown("---")
            st.subheader("5. Data Export")
            export_dict = {'Time': time_arr, f'Raw_{signal_choice}': raw_signal}
            for method in methods: export_dict[f'{method}_Filtered'] = results[method]
            df_export = pd.DataFrame(export_dict)
            csv_data = df_export.to_csv(index=False).encode('utf-8')
            st.download_button(label=f"Download full data ({total_points:,} points .csv)", data=csv_data, file_name='post_processing_data.csv', mime='text/csv')

    else:
        st.info("Please upload a data file in the left sidebar to begin.")

if __name__ == '__main__':
    if st.runtime.exists():
        main()
    else:
        from streamlit.web import cli
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(cli.main())