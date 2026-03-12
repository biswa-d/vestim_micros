"""
Filter-based padding calculation utilities.

Determines optimal padding length based on filter characteristics.
Padding is applied before filtering to avoid filter edge effects where
initial filtered values start at zero rather than the actual signal range.
"""

import math


class FilterPaddingCalculator:
    """Calculate padding length based on filter time constants."""
    
    # Filter type coefficients (empirically determined)
    # These represent multiples of filter time constant to eliminate edge effects
    FILTER_COEFFICIENTS = {
        'butterworth': 5.0,      # 5x time constant
        'lowpass': 5.0,
        'moving_average': 3.0,   # 3x window size (treated as time constant)
        'savitzky_golay': 2.0,   # Shorter; less edge effect sensitivity
        'exponential_moving_average': 8.0,  # Longer tail; needs more padding
    }
    
    @staticmethod
    def calculate_time_constant(filter_config):
        """
        Calculate the time constant of a filter based on its configuration.
        
        Time constant τ represents how quickly the filter settles to steady state.
        For different filters:
        - Butterworth/Lowpass: τ = 1 / (2π * f_c) where f_c is cutoff frequency in Hz
        - Moving Average: τ ≈ window_size / 2
        - Savitzky-Golay: τ ≈ window_size / 2
        - EMA: τ = dt * (1 - α) / α where α is smoothing factor
        
        :param filter_config: Dictionary with filter parameters
        :return: Time constant in samples
        """
        filter_type = filter_config.get('type', '').lower()
        
        if 'butterworth' in filter_type or 'lowpass' in filter_type:
            # τ = 1 / (2π * f_c) in seconds, then multiply by sampling rate to get samples
            corner_freq = filter_config.get('corner_frequency', 0.02)  # in Hz
            sampling_rate = filter_config.get('sampling_rate', 1.0)    # in Hz
            
            if corner_freq > 0:
                time_const_seconds = 1.0 / (2.0 * math.pi * corner_freq)
                time_const_samples = time_const_seconds * sampling_rate
                return max(1, int(time_const_samples))
            return 10  # Default
        
        elif 'moving_average' in filter_type:
            # For moving average, window size is the time constant
            window_size = filter_config.get('window_size', 5)
            return max(1, int(window_size / 2))
        
        elif 'savitzky' in filter_type:
            # For Savitzky-Golay, window size is less critical for edge effects
            window_size = filter_config.get('window_size', 5)
            return max(1, int(window_size / 2))
        
        elif 'exponential' in filter_type or 'ema' in filter_type:
            # For EMA: τ = dt * (1 - α) / α
            alpha = filter_config.get('alpha', 0.3)
            sampling_rate = filter_config.get('sampling_rate', 1.0)  # dt in seconds
            
            if alpha > 0 and alpha < 1:
                time_const_samples = (1.0 - alpha) / alpha
                return max(1, int(time_const_samples))
            return 20  # Default for EMA
        
        return 10  # Default fallback
    
    @staticmethod
    def calculate_padding_length(filters, sampling_rate=1.0, multiplier_override=None):
        """
        Calculate total padding length needed for a set of filters.
        
        The padding length is the maximum time constant across all filters,
        multiplied by a filter-type-specific coefficient to account for
        settling time to steady state.
        
        :param filters: List of filter configuration dicts
        :param sampling_rate: Sampling frequency in Hz
        :param multiplier_override: Optional override for the coefficient multiplier
        :return: Padding length in samples (integer)
        """
        
        if not filters:
            return 0
        
        max_time_const = 0
        max_filter_type = None
        
        for filt in filters:
            filt_config = filt.copy()
            filt_config['sampling_rate'] = sampling_rate
            
            time_const = FilterPaddingCalculator.calculate_time_constant(filt_config)
            
            if time_const > max_time_const:
                max_time_const = time_const
                max_filter_type = filt.get('type', '').lower()
        
        # Get the coefficient for this filter type
        if multiplier_override is not None:
            coeff = multiplier_override
        else:
            # Find matching coefficient
            coeff = 10.0  # Default
            for key, value in FilterPaddingCalculator.FILTER_COEFFICIENTS.items():
                if key in (max_filter_type or ''):
                    coeff = value
                    break
        
        # Total padding = time_constant * coefficient
        padding_length = int(max_time_const * coeff)
        
        # Ensure at least some padding if filters are present
        if filters and padding_length < 1:
            padding_length = max(10, int(max_time_const * 3))
        
        return padding_length
    
    @staticmethod
    def calculate_padding_for_test_data(job_metadata):
        """
        Calculate padding needed based on augmentation metadata from a job folder.
        
        :param job_metadata: Dictionary from job_metadata.json or augmentation_metadata.json
        :return: Recommended padding length in samples
        """
        
        # Check for augmentation metadata (preferred)
        applied_filters = job_metadata.get('applied_filters', [])
        if applied_filters:
            sampling_rate = job_metadata.get('sampling_rate', 1.0)
            return FilterPaddingCalculator.calculate_padding_length(applied_filters, sampling_rate)
        
        # Fallback: check for filter settings from augmentation_settings
        filter_settings = job_metadata.get('filter_settings', {})
        if filter_settings and filter_settings.get('apply_filter'):
            filters = []
            if filter_settings.get('filter_type'):
                filters.append({
                    'type': filter_settings.get('filter_type'),
                    'corner_frequency': filter_settings.get('corner_frequency'),
                    'sampling_rate': filter_settings.get('sampling_rate', 1.0),
                    'order': filter_settings.get('order', 1)
                })
            sampling_rate = filter_settings.get('sampling_rate', 1.0)
            return FilterPaddingCalculator.calculate_padding_length(filters, sampling_rate)
        
        return 0


# Example usage:
if __name__ == "__main__":
    # Example 1: Butterworth 0.02 Hz LPF at 1 Hz sampling
    filters = [
        {
            'type': 'butterworth',
            'corner_frequency': 0.02,
            'sampling_rate': 1.0,
            'order': 1
        }
    ]
    
    pad_len = FilterPaddingCalculator.calculate_padding_length(filters, sampling_rate=1.0)
    print(f"Calculated padding: {pad_len} samples")
    # For 0.02 Hz cutoff: τ = 1/(2π*0.02) ≈ 7.96 seconds = ~8 samples at 1 Hz
    # With 5x multiplier: 8 * 5 = 40 samples
    
    # Example 2: Multiple filters (max wins)
    filters = [
        {'type': 'butterworth', 'corner_frequency': 0.02, 'sampling_rate': 1.0},
        {'type': 'moving_average', 'window_size': 50}
    ]
    
    pad_len = FilterPaddingCalculator.calculate_padding_length(filters, sampling_rate=1.0)
    print(f"Calculated padding for multiple filters: {pad_len} samples")
