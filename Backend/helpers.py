import numpy as np

def scale_lat_long(lat_long_array, scaler):
    """
    Scales latitude and longitude using a fitted scaler.

    Parameters:
        lat_long_array (np.ndarray): A 2D array with [[latitude, longitude]].
        scaler (sklearn.preprocessing.StandardScaler): Fitted scaler for lat/long.

    Returns:
        Tuple of (scaled_lat_long_array, scaler)
    """
    if not isinstance(lat_long_array, np.ndarray):
        raise ValueError("Input must be a numpy array")
    if scaler is None:
        raise ValueError("Scaler is not provided")
    return scaler.transform(lat_long_array), scaler
