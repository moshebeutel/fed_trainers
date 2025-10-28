import gc
from pathlib import Path
from typing import Optional
import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import librosa

def stft(signal: torch.Tensor, n_fft: int=4096, hop_length: int=1024, win_length: Optional[int]=None):
    """
    Computes the Short-Time Fourier Transform (STFT) of the input signal.

    This function calculates the STFT of a given signal using the specified
    parameters for the FFT size, hop length, and window length. The STFT is a
    representation of the signal in the time-frequency domain, commonly used
    in various signal processing applications.

    Args:
        signal (torch.Tensor): The input signal tensor to be transformed.
        n_fft (int): The size of the FFT to be applied. Determines the frequency
            resolution of the STFT. Defaults to 4096.
        hop_length (int): The number of samples to advance between successive
            frames. Defaults to 1024.
        win_length (Optional[int]): The size of the window to be applied to each
            frame. If None, defaults to the value of `n_fft`.

    Returns:
        torch.Tensor: The resulting STFT tensor computed from the input signal.
    """
    if win_length is None:
        win_length = n_fft

    stft_tensor = torch.stft(
        signal,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=torch.hann_window(win_length).to(signal.device),
        return_complex=True
    )

    return stft_tensor


def estimate_fundamental_frequency_torch(signal: torch.Tensor, fs: int, n_fft: int=4096, hop_length: int=1024, win_length: Optional[int]=None):
    """
    Estimates the fundamental frequency of a given signal using the Short-Time Fourier Transform (STFT).

    The function calculates the Short-Time Fourier Transform (STFT) of the input signal using the specified
    parameters, computes the power spectrum, and identifies the frequency bin with the maximum power. It
    returns the estimated fundamental frequency, the corresponding frequency bin, and the full STFT tensor.

    Args:
        signal: A 1D tensor representing the input audio signal.
        fs: The sampling rate of the signal in Hz.
        n_fft: The size of the FFT window.
        hop_length: The number of audio samples between adjacent STFT columns.
        win_length: The size of the window applied to each frame. Defaults to the value of `n_fft` if None.

    Returns:
        Tuple[float, int, Tensor]: A tuple containing:
            - f0: The estimated fundamental frequency in Hz.
            - fund_bin: The index of the frequency bin corresponding to the fundamental frequency.
            - stft_tensor: The computed STFT tensor for the input signal.
            - power_spectrum: The power spectrum of the STFT tensor.
            - mean_power: The mean power of the power spectrum.
            - magnitudes: The magnitudes of the STFT tensor.
    """

    stft_tensor = stft(signal, n_fft, hop_length, win_length)

    magnitudes = torch.abs(stft_tensor)
    power_spectrum = magnitudes ** 2
    mean_power = power_spectrum.mean(dim=1)

    fund_bin = torch.argmax(mean_power).item()
    f0 = fund_bin * fs / n_fft

    return f0, fund_bin, stft_tensor, power_spectrum, mean_power, magnitudes



def torch_unwrap(p, discontinuity=np.pi):
    """
    Unwraps a phase tensor by correcting discontinuities.

    This function removes phase discontinuities by assuming a default wrapping
    range of `[-pi, pi]` unless a different range is specified through the
    `discontinuity` parameter. The correction is applied sequentially along the last
    dimension of the input tensor.

    Args:
        p (torch.Tensor): A tensor of phase values to be unwrapped.
            It is expected to have at least one trailing dimension.
        discontinuity (float, optional): The discontinuity threshold for phase wrapping.
            Defaults to pi, meaning phase values are wrapped to the range [-pi, pi].

    Returns:
        torch.Tensor: The unwrapped version of the input phase tensor, where
        discontinuities exceeding the specified threshold have been corrected.
    """
    dp = p[..., 1:] - p[..., :-1]
    dd = torch.remainder(dp + np.pi, 2 * np.pi) - np.pi
    dd = torch.where((dd == -np.pi) & (dp > 0), np.pi, dd)

    ph_corrected = torch.cat((p[..., :1], p[..., 1:] - (dp - dd).cumsum(dim=-1)), dim=-1)
    return ph_corrected


def estimate_frequency_from_phase_diff(stft_tensor, fund_bin, fs, hop_length, plot_freq_diff=True):
    """
    Estimates the fundamental frequency of a signal from its short-time Fourier transform (STFT)
    representation using the phase difference method. It first calculates the phase at a specified
    frequency bin, then computes the difference between successive phases to infer the frequency,
    and finally computes the mean frequency as an estimation. The method generates a plot showing
    the frequency estimates for each frame and the average frequency.

    Args:
        stft_tensor (Tensor): Complex-valued tensor representing the STFT of the input signal.
        fund_bin (int): Index of the frequency bin corresponding to the fundamental frequency.
        fs (float): Sampling rate of the input signal in Hz.
        hop_length (int): Hop length used in the STFT computation.
        plot_freq_diff (bool, optional): Whether to plot the frequency difference between successive
    Returns:
        float: Mean estimated fundamental frequency (f₀) in Hz over all time frames.

    """

    phase = torch.angle(stft_tensor[fund_bin])

    dphi = torch.diff(phase)
    dphi = torch_unwrap(dphi)  # remove 2*pi multiplicities

    # compute freq using phase
    f0_est = (dphi / (2 * torch.pi)) * (fs / hop_length)

    # mean of freq.
    f0_mean = torch.mean(f0_est)

    # plot
    if plot_freq_diff:
        plt.figure(figsize=(10, 3))
        plt.plot(f0_est.cpu(), label='Estimated f₀ per frame')
        plt.axhline(f0_mean.cpu(), color='r', linestyle='--', label=f'Mean f₀ ≈ {f0_mean:.2f} Hz')
        plt.title(f'Phase-diff Frequency Estimate (bin {fund_bin})')
        plt.xlabel('Time Frame')
        plt.ylabel('Frequency [Hz]')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        # plt.show()
        plt.savefig(f'images/phase_diff_freq_bin{fund_bin}.png')
        plt.close()

    return f0_mean


def refine_peak_frequency_parabolic(mean_power, fs, n_fft):
    """
    Refines the peak frequency by performing parabolic interpolation on a power spectrum.
    The function attempts to improve the granularity of the frequency estimation by calculating
    a sub-bin shift using the values around the detected peak.

    Args:
        mean_power (torch.Tensor): A tensor representing the power spectrum, typically derived
            from the Fourier coefficients. Each value corresponds to the signal energy at a
            specific frequency bin.
        fs (float): The sampling frequency of the signal from which the spectrum was computed,
            used to scale the bin number to actual frequency values.
        n_fft (int): The size of the FFT (Fast Fourier Transform), used to calculate the proper
            scaling between bin indices and the actual frequency in Hz.

    Returns:
        Tuple[float, float]: A tuple containing the refined frequency in Hz and the refined
        bin index. If the peak bin is at the edge of the power spectrum (0 or len(mean_power)-1),
        no interpolation is performed, and the original peak frequency and bin index are returned.
    """
    peak_bin = torch.argmax(mean_power).item()
    if peak_bin == 0 or peak_bin == len(mean_power) - 1:
        return peak_bin * fs / n_fft, peak_bin

    alpha = mean_power[peak_bin - 1].item()
    beta  = mean_power[peak_bin].item()
    gamma = mean_power[peak_bin + 1].item()

    p = 0.5 * (alpha - gamma) / (alpha - 2 * beta + gamma)
    refined_bin = peak_bin + p
    refined_freq = refined_bin * fs / n_fft
    return refined_freq, refined_bin


def analyze_single_tone_stft(signal: torch.Tensor, fs: int, n_fft: int=8192, hop_length: Optional[int]=1024,
                             win_length: Optional[int]=None, title: str='', plot_power=True, plot_magnitude=True,
                             plot_phase=True,
                             plot_phase_diff=True):
    """
    Analyzes a single-tone signal using Short-Time Fourier Transform (STFT) to extract
    frequency characteristics such as coarse and refined fundamental frequencies, and
    to visualize the power spectrum and phase spectrum.

    Args:
        signal (torch.Tensor): Input signal to analyze. Should be a 1D tensor representing
            the waveform.
        fs (int): Sampling frequency of the input signal in Hz.
        n_fft (int): Number of FFT points to use for the STFT. Determines the frequency
            resolution.
        hop_length (Optional[int]): Number of samples between successive STFT frames. If not
            specified, defaults to 1024.
        win_length (Optional[int]): Length of window for each segment in STFT. If not provided,
            defaults to `n_fft`.
        title (str): Title or identifier for the generated plots and outputs.
        plot_power (bool): Whether to plot the power spectrum. Defaults to True.
        plot_magnitude (bool): Whether to plot the magnitude spectrum. Defaults to True.
        plot_phase (bool): Whether to plot the phase spectrum. Defaults to True.
        plot_phase_diff (bool): Whether to plot the phase difference between successive bins.

    Returns:
        dict: A dictionary containing analysis results:
            - refined_f0 (float): Refined fundamental frequency (f₀) in Hz.
            - coarse_f0 (float): Approximate f₀ calculated from the bin with maximum power.
            - phase_f0 (float): Estimated f₀ computed from phase difference.
            - bin_index (int): Index of the spectral bin corresponding to coarse f₀.
            - refined_bin (float): Refined bin index determined through interpolation.
            - stft (torch.Tensor): STFT result as a complex-valued 2D tensor with time frames
              along one axis and frequency bins along the other.

    """

    # coarse estimation
    f0, fund_bin, stft_tensor, power_spectrum, mean_power, magnitudes = estimate_fundamental_frequency_torch(signal, fs, n_fft, hop_length, win_length)

    # parabolic interpolation
    refined_freq, refined_bin = refine_peak_frequency_parabolic(mean_power, fs, n_fft)

    stft_means = torch.mean(stft_tensor, dim=1)

    harmonics = [1, 2, 3, 4, 5]

    t_harmonics = librosa.interp_harmonics(stft_means.cpu().numpy(),
                                           freqs=np.arange(len(stft_tensor)),
                                           harmonics=harmonics, axis=0)


    # plot power stft
    if plot_power:
        power_db = 10 * torch.log10(power_spectrum + 1e-12)
        norm_mode = 'mean+3std'
        # Flatten to 1D for normalization
        power_flat = power_db[0].flatten()

        if norm_mode == "max-40dB":
            vmax = power_flat.max().item()
            vmin = vmax - 40
        elif norm_mode == "mean+3std":
            mean = power_flat.mean().item()
            std = power_flat.std().item()
            vmin = mean
            vmax = mean + 3 * std
        elif norm_mode == "percentile":
            sorted_vals = torch.sort(power_flat).values
            vmin = sorted_vals[int(0.05 * len(sorted_vals))].item()
            vmax = sorted_vals[int(0.95 * len(sorted_vals))].item()
        else:
            vmin = power_flat.min().item()
            vmax = power_flat.max().item()


        plt.figure(figsize=(12, 4))
        plt.imshow(power_db.cpu(), origin='lower', aspect='auto', cmap='viridis' , vmin=vmin, vmax=vmax)
        plt.axhline(refined_bin, xmax=0.2, color='r', linestyle='--', label=f'f₀ ≈ {refined_freq:.2f} Hz')
        plt.title(f'{title} – STFT Power + refined f₀')
        plt.xlabel('Time Frame')
        plt.ylabel('Frequency Bin')
        plt.colorbar(label='Power (dB)')
        plt.legend()
        plt.tight_layout()
        # plt.show()
        plt.savefig(f'images/{title}_power_refined.png')
        plt.close()

    # plot phase stft
    if plot_phase:
        phase = torch.angle(stft_tensor)
        plt.figure(figsize=(12, 4))
        plt.imshow(phase.cpu(), origin='lower', aspect='auto', cmap='twilight')
        plt.axhline(refined_bin, color='r', linestyle='--', label=f'f₀ ≈ {refined_freq:.2f} Hz')
        plt.title(f'{title} – STFT Phase + refined f₀')
        plt.xlabel('Time Frame')
        plt.ylabel('Frequency Bin')
        plt.colorbar(label='Phase (radians)')
        plt.legend()
        plt.tight_layout()
        # plt.show()
        plt.savefig(f'images/{title}_phase_refined.png')
        plt.close()

    # freq from phase
    f0_from_phase = estimate_frequency_from_phase_diff(stft_tensor, fund_bin, fs, hop_length, plot_freq_diff=plot_phase_diff)

    return {
        'refined_f0': refined_freq,
        'coarse_f0': f0,
        'phase_f0': f0_from_phase,
        'bin_index': fund_bin,
        'refined_bin': refined_bin,
        'stft': stft_tensor.cpu()
    }


def analyze_rf_spectrum(signal: torch.Tensor, fs: float, f_center: float, title: str= 'RF Spectrum', plot_spectrum=True):
    """
    Compute the radio frequency (RF) spectrum of a given signal and plot if specified.

    This function computes the FFT of the given signal to derive its spectrum in
    the frequency domain. It then shifts the spectrum to adjust for RF frequencies
    considering the central frequency. The result is visualized on a plot,
    displaying the power levels of the signal relative to its frequency, and the
    plot is saved to a file with a specified title.

    Args:
        signal (torch.Tensor): Input signal in the time domain.
        fs (float): Sampling frequency of the signal in Hz.
        f_center (float): Central frequency of the RF signal in Hz.
        title (str): Title of the plot, also used for naming the saved file.
        plot_spectrum (bool): Whether to plot the spectrum. Defaults to True.

    Returns:
        freqs_rf_mhz (torch.Tensor): Frequencies of the RF spectrum in MHz.
        power_db (torch.Tensor): Power levels of the RF spectrum in dB.
    """
    n = len(signal)

    fft_vals = torch.fft.fftshift(torch.fft.fft(signal, n=n))
    fft_freqs = torch.fft.fftshift(torch.fft.fftfreq(n, d=1 / fs))

    # power (dB)
    power_db = 10 * torch.log10(torch.abs(fft_vals) ** 2 + 1e-12)

    # rf absolute values
    freqs_rf = fft_freqs + f_center
    freqs_rf_mhz = freqs_rf / 1e6


    # plot
    if plot_spectrum:
        plt.figure(figsize=(10, 4))
        plt.plot(freqs_rf_mhz.cpu(), power_db.cpu(), label='RF Spectrum')
        plt.title(title)
        plt.xlabel('Frequency [MHz]')
        plt.ylabel('Power [dB]')
        plt.grid(True)
        plt.tight_layout()
        # plt.show()
        plt.savefig(f'images/{title}_absolute.png')
        plt.close()

    return freqs_rf_mhz, power_db

def load_complex_signal(h5_file, signal_name: str, trim:Optional[tuple[float, float]]=None, real_idx=0, imag_idx=1, device='cuda') -> torch.Tensor:
    """
    Loads a complex signal from an HDF5 file, prepares it for further use, and returns it as a
    PyTorch tensor. The signal is split into real and imaginary parts specified by their indices,
    and has an optional trim functionality based on proportions of the signal's span.

    Args:
        h5_file: The HDF5 file object containing the signal data.
        signal_name (str): The name of the dataset within the HDF5 file that holds the signal.
        trim (Optional[tuple[float, float]]): A tuple specifying the start and end proportions
            of the signal to be retained. Must be values between 0 and 1, where the first
            value is less than the second.
        real_idx (int): The index of the row corresponding to the real part of the signal
            in the dataset.
        imag_idx (int): The index of the row corresponding to the imaginary part of the signal
            in the dataset.
        device (str): The device on which the tensor should be loaded (e.g., 'cuda' or 'cpu').

    Returns:
        torch.Tensor: A tensor representing the complex signal, where the real and
        imaginary parts have been combined into a single complex-valued tensor.
    """

    signal = torch.from_numpy(h5_file[f'{signal_name}'][:]).to(device)
    m,n = signal.shape
    assert m == 2, f'Expected 2 rows for real and imaginary'
    if trim is not None:
        assert isinstance(trim, tuple), f'Expected tuple for trim'
        assert len(trim) == 2, f'Expected 2 elements for trim'
        assert trim[0] < trim[1], f'Expected trim[0] < trim[1]'
        assert trim[0] >= 0 and trim[1] <= 1, f'Expected trim[0], trim[1] in [0, 1]'
        signal = signal[:, int(trim[0] * n):int(trim[1] * n)]

    real = signal[real_idx, :]
    imag = signal[imag_idx, :]

    return real + 1j * imag

def load_mat(file_path, names=None, trim:Optional[tuple[float, float]]=None):
    """
    Loads signals from a .mat file, filtering by specified signal names and trimming if needed.

    This function reads a MATLAB .mat file, retrieves signals listed under specified names, and
    returns them. If no names are specified, a default set of signal names will be used. Optionally,
    a trimming range can be applied to the signals.

    Args:
        file_path (str or Path): The path to the .mat file to load signals from. The file must exist
            and have a .mat extension.
        names (list[str] or None): A list of signal names to load from the .mat file. If None, it
            defaults to ['Sine', 'Nojamming', 'Gaussian'].
        trim (Optional[tuple[float, float]]): An optional tuple specifying the range for trimming
            the signals. If provided, it trims the signals to the range (start, end).

    Returns:
        Union[tuple, Any]: If multiple signals are loaded, a tuple of the loaded signals is returned.
            If only a single signal is loaded, that signal is returned directly.
    """
    if names is None:
        names = ['Sine', 'Nojamming', 'Gaussian']
    w1_mat = Path(file_path)
    assert w1_mat.exists(), f'File not found: {w1_mat}'
    assert w1_mat.suffix == '.mat', f'Expected .mat file, got {w1_mat.suffix}'
    # load mat file
    with h5py.File(w1_mat, 'r') as f:

        hf_signal_names = []

        # print signal names
        def print_h5_structure(name, obj):
            print(name)
            hf_signal_names.append(name)

        f.visititems(print_h5_structure)

        signals_loaded = {}
        for name in hf_signal_names:

            if name not in names:
                print(f'Skipping {name}')
            else:
                print(f'Loading {name}')
                sig = load_complex_signal(f, signal_name=name, trim=trim)
                signals_loaded[name] = sig

    return tuple(signals_loaded.values()) if len(signals_loaded) > 1 else signals_loaded[names[0]]

if __name__ == "__main__":

    sine = load_mat(file_path='/home/user1/Downloads/w1.mat', names=['Sine'], trim=(0.55, 0.6))

    result = analyze_single_tone_stft(
        signal=sine,
        fs=1_000_000,
        n_fft=256,
        hop_length=128,
        title='Sine Signal Analysis'
    )

    print(f"Coarse f₀:   {result['coarse_f0']:.2f} Hz (bin {result['bin_index']})")
    print(f"Refined f₀:  {result['refined_f0']:.2f} Hz (bin {result['refined_bin']:.3f})")
    print(f"Phase f₀:    {result['phase_f0']:.2f} Hz")

    freqs_rf_mhz, power_db = analyze_rf_spectrum(
        signal=sine,
        fs=1_000_000,  # 1 Msps
        f_center=900_000_000,  # 900 MHz
        title='Sine Signal – RF Spectrum'
    )

    print(f'RF Freq. (MHz) range : {freqs_rf_mhz.min():.2f} - {freqs_rf_mhz.max():.2f} mean {freqs_rf_mhz.mean():.2f} std {freqs_rf_mhz.std():.2f}')
    print(f'RF Power (dB) range : {power_db.min():.2f} - {power_db.max():.2f} mean {power_db.mean():.2f} std {power_db.std():.2f}')