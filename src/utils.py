"""Utility functions for OFDM channel estimation."""
import re
import torch

def extract_values(file_name):
    """
    Extract channel information from a file name.

    Parses file names with format:
    '{number}_SNR-{snr}_DS-{delay_spread}_DOP-{doppler}_N-{pilot_freq}_{channel_type}.mat'

    Args:
        file_name: The file name from which to extract channel information

    Returns:
        tuple: A tuple containing:
            - file_number (torch.Tensor): The file number
            - snr (torch.Tensor): Signal-to-noise ratio value
            - delay_spread (torch.Tensor): Delay spread value
            - max_doppler_shift (torch.Tensor): Maximum Doppler shift value
            - pilot_placement_frequency (torch.Tensor): Pilot placement frequency
            - channel_type (list): The channel type

    Raises:
        ValueError: If the file name does not match the expected pattern
    """
    pattern = r'(\d+)_SNR-(\d+)_DS-(\d+)_DOP-(\d+)_N-(\d+)_([A-Z\-]+)\.mat'
    match = re.match(pattern, file_name)
    if match:
        file_no = torch.tensor([int(match.group(1))], dtype=torch.float)
        snr_value = torch.tensor([int(match.group(2))], dtype=torch.float)
        ds_value = torch.tensor([int(match.group(3))], dtype=torch.float)
        dop_value = torch.tensor([int(match.group(4))], dtype=torch.float)
        n = torch.tensor([int(match.group(5))], dtype=torch.float)
        channel_type = [match.group(6)]
        return file_no, snr_value, ds_value, dop_value, n, channel_type
    else:
        raise ValueError("Cannot extract file information.")

def concat_complex_channel(channel_matrix):
    """
    Convert a complex channel matrix into a real matrix by concatenating real and imaginary parts.

    Transforms a complex tensor into a real-valued tensor by concatenating
    the real and imaginary components along the specified dimension.

    Args:
        channel_matrix: Complex channel matrix

    Returns:
        Real-valued channel matrix with concatenated real and imaginary parts
    """
    real_channel_m = torch.real(channel_matrix)
    imag_channel_m = torch.imag(channel_matrix)
    cat_channel_m = torch.cat((real_channel_m, imag_channel_m), dim=1)
    return cat_channel_m
