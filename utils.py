from collections import OrderedDict
import math

import scipy.signal as signal
import numpy as np

def compute_grad_norm(model):
    norm = 0.0
    for p in model.parameters():
        cur_grad = p.grad.square().sum() if p.grad is not None else 0
        norm += cur_grad
    
    return math.sqrt(norm)


def compute_weight_norm(model):
    norm = 0.0
    #norm_dict = OrderedDict()
    for n, p in model.named_parameters():
        cur_weight_norm = p.square().sum().cpu()
        #norm_dict[n] = cur_weight_norm.sqrt().cpu()
        norm += cur_weight_norm
    
    return math.sqrt(norm)


def compute_stats_by_last_dim(t):
    d = {}
    d["min"] = t.min(dim=-1).values.mean().item()
    d["max"] = t.max(dim=-1).values.mean().item()
    d["mean"] = t.mean(dim=-1).mean().item()
    d["std"] = t.std(dim=-1).mean().item()
    return d



def bandwidth_extension_sbr_transposed(input_signal, fs):
    # Parameters
    n_fft = 512
    hop_length = n_fft // 4
    win_length = n_fft
    window = 'hann'
    
    # Compute STFT
    f, t, Zxx = signal.stft(input_signal, fs, window=window, nperseg=win_length,
                            noverlap=win_length - hop_length, nfft=n_fft)
    
    # Identify frequency bins
    freq_resolution = fs / n_fft
    low_freq_limit = 50    # Minimum frequency to consider
    high_freq_limit = 4000  # Maximum frequency of the input signal
    extended_freq_limit = 8000  # Desired maximum frequency after BWE
    
    # Frequency bin indices
    k_low = int(low_freq_limit / freq_resolution)
    k_high = int(high_freq_limit / freq_resolution)
    k_extended = int(extended_freq_limit / freq_resolution)
    
    # Magnitude and phase
    magnitude = np.abs(Zxx)
    phase = np.angle(Zxx)
    
    # Initialize extended magnitude and phase
    extended_magnitude = np.copy(magnitude)
    extended_phase = np.copy(phase)
    
    # Spectral replication with transposition
    num_bins_to_replicate = k_high - k_low
    k_shift = k_high  # Amount to shift in frequency bins
    
    for i in range(num_bins_to_replicate):
        source_bin = k_low + i
        target_bin = k_shift + i
        if target_bin < k_extended:
            # Transpose the spectrum by flipping the order
            transposed_bin = k_high - i - 1  # Subtract 1 to adjust index
            if transposed_bin >= k_low:
                # Replicate magnitude with transposition
                extended_magnitude[target_bin, :] = magnitude[transposed_bin, :]
                # Assign random phase to replicated components
                extended_phase[target_bin, :] = np.random.uniform(-np.pi, np.pi, size=phase.shape[1])
                # Spectral shaping (apply decay)
                decay = np.exp(-0.5 * ((f[target_bin] - high_freq_limit) / (extended_freq_limit - high_freq_limit)) ** 2)
                extended_magnitude[target_bin, :] *= decay
    
    # Zero out frequencies above extended_freq_limit
    extended_magnitude[k_extended:, :] = 0
    extended_phase[k_extended:, :] = 0
    
    # Reconstruct extended complex spectrum
    extended_Zxx = extended_magnitude * np.exp(1j * extended_phase)
    
    # Inverse STFT
    _, extended_signal = signal.istft(extended_Zxx, fs, window=window, nperseg=win_length,
                                      noverlap=win_length - hop_length, nfft=n_fft)
    
    # Combine with original signal (optional scaling)
    alpha = 0.5  # Adjust based on perceptual quality
    extended_signal = input_signal[:len(extended_signal)] + alpha * extended_signal
    
    # Normalize to prevent clipping
    extended_signal /= np.max(np.abs(extended_signal))
    
    return extended_signal