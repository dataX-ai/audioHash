import librosa
import numpy as np
import matplotlib.pyplot as plt
import time
from skimage.feature import peak_local_max  # Move import to top level
import struct
import time
from bitarray import bitarray
from scipy.optimize import linear_sum_assignment

class AudioHash:
    """
    A highly efficient binary encoder/decoder for audio fingerprint peaks.
    Uses bit packing to minimize size while preserving precision.
    
    Each peak is encoded as:
    - Frequency (0-1): 10 bits (0-1023 values, ~0.001 precision)
    - Time:
        - Minutes: 6 bits (0-63 minutes)
        - Seconds within minute: 12 bits (0-4095 values, ~15ms precision)
    - Amplitude (0-1): 10 bits (0-1023 values, ~0.001 precision)
    
    Total: 38 bits per peak (4.75 bytes)
    """
    
    # Constants for bit allocation
    FREQ_BITS = 10
    MINUTE_BITS = 6
    SECOND_BITS = 12  # Increased from 10 to 12 for better precision
    AMP_BITS = 10
    BITS_PER_PEAK = FREQ_BITS + MINUTE_BITS + SECOND_BITS + AMP_BITS
    
    # Max values for each component
    FREQ_MAX = 2**FREQ_BITS - 1    # 1023
    MINUTE_MAX = 2**MINUTE_BITS - 1  # 63 minutes
    SECOND_MAX = 2**SECOND_BITS - 1  # 4095 (for 0-59.94 seconds)
    AMP_MAX = 2**AMP_BITS - 1      # 1023
    
    # Scale factors for numeric conversion
    FREQ_SCALE = 1000       # Allows 0.000 to 1.000 with 0.001 precision
    SECOND_SCALE = 68.31    # Approx 4095/60 to map 0-59.94 seconds to 0-4095 range
    AMP_SCALE = 1000        # Allows 0.000 to 1.000 with 0.001 precision
    
    def __init__(self):
        """Initialize the encoder/decoder"""
        pass
    
    def create_time_sensitive_hash(self, audio_path, num_peaks=256, visualize=False):
        """
        Create a time-sensitive hash based on the exact positions of prominent peaks.
        This hash requires all peaks to match closely for audio to be considered the same.
        
        Args:
            audio_path: Path to the audio file
            num_peaks: Number of peaks to detect
            visualize: Whether to visualize the peaks
            
        Returns:
            A list of (freq, time, normalized_amplitude) tuples representing the hash
        """
        
        # Load audio file with mono=True to speed up loading and processing
        audio, sr = librosa.load(audio_path, sr=None, mono=True)
        
        # Use shorter audio for hashing if it's very long (optional)
        # max_duration = 60  # seconds
        # if len(audio) > max_duration * sr:
        #     audio = audio[:max_duration * sr]
        
        # Compute mel spectrogram with higher resolution
        n_mels = 128  # Higher mel resolution
        hop_length = 512  # Default hop length in librosa
        
        # Pre-compute n_fft for better performance
        n_fft = 2048
        mel_spec = librosa.feature.melspectrogram(
            y=audio, 
            sr=sr, 
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length,
            fmax=sr/2.0  # Explicitly set fmax for better performance
        )
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Adaptive threshold approach to ensure we get enough peaks
        threshold = -30  # Start with this threshold
        min_distance = 3  # Start with this minimum distance
        max_attempts = 10  # Increased number of attempts
        
        for attempt in range(max_attempts):
            coordinates = peak_local_max(
                mel_db, 
                min_distance=min_distance,
                threshold_abs=threshold,
                num_peaks=num_peaks*3  # Get more peaks than needed for filtering
            )
            
            
            if len(coordinates) >= num_peaks:
                # We found enough peaks
                break
            
            # If we don't have enough peaks, make threshold more permissive
            threshold -= 10  # Make threshold less strict (e.g., -30 -> -40 -> -50)
            
            # If threshold becomes too permissive, adjust min_distance instead
            if threshold < -80:
                threshold = -30  # Reset threshold
                min_distance = max(1, min_distance - 1)  # Reduce min_distance (but keep it ≥ 1)
        
        # Extract peak values for the coordinates we found
        if len(coordinates) > 0:
            # Get the values at those coordinates
            peak_values = mel_db[coordinates[:, 0], coordinates[:, 1]]
            
            # Sort peaks by amplitude (descending)
            sort_idx = np.argsort(peak_values)[::-1]
            coordinates = coordinates[sort_idx]
            peak_values = peak_values[sort_idx]            
        else:
            # No peaks found at all
            peak_values = np.array([])

        
        # Prepare final coordinates and peak values
        if len(coordinates) > num_peaks:
            # Take only the top num_peaks
            final_coordinates = coordinates[:num_peaks]
            final_peak_values = peak_values[:num_peaks]
        elif len(coordinates) < num_peaks:
            # Pad with zeros to reach num_peaks
            num_missing = num_peaks - len(coordinates)
            
            # If we have some peaks, use them
            if len(coordinates) > 0:
                final_coordinates = coordinates
                final_peak_values = peak_values
            else:
                # If no peaks, create empty arrays
                final_coordinates = np.zeros((0, 2), dtype=np.int64)
                final_peak_values = np.array([])
            
            # Create zero padding
            zero_coords = np.zeros((num_missing, 2), dtype=np.int64)
            zero_values = np.full(num_missing, -80)  # Use DB_MIN for zero pad values
            
            # Combine real peaks with zero padding
            final_coordinates = np.vstack((final_coordinates, zero_coords)) if len(final_coordinates) > 0 else zero_coords
            final_peak_values = np.concatenate((final_peak_values, zero_values)) if len(final_peak_values) > 0 else zero_values
        else:
            # Exactly right number of peaks
            final_coordinates = coordinates
            final_peak_values = peak_values
        
        # Constants for normalization
        DB_MIN = -80  # Typical minimum audible dB value
        DB_MAX = 0    # Maximum dB value (when ref=np.max is used)
        
        # Calculate time in seconds for each peak
        if len(final_coordinates) > 0:
            times_seconds = librosa.frames_to_time(
                final_coordinates[:, 1], 
                sr=sr, 
                hop_length=hop_length
            )
        else:
            times_seconds = np.zeros(num_peaks)
        
        # Clip and normalize values
        peak_values_clipped = np.clip(final_peak_values, DB_MIN, DB_MAX)
        normalized_amplitudes = (peak_values_clipped - DB_MIN) / (DB_MAX - DB_MIN)
        
        # Create list of tuples
        if len(final_coordinates) > 0:
            norm_freqs = final_coordinates[:, 0] / n_mels
        else:
            norm_freqs = np.zeros(num_peaks)
        
        peaks = list(zip(norm_freqs, times_seconds, normalized_amplitudes))
        
        # Sort peaks by time to maintain temporal order
        peaks.sort(key=lambda x: x[1])
        
        assert len(peaks) == num_peaks, f"Expected {num_peaks} peaks, got {len(peaks)}"
        
        # Visualization
        if visualize:
            plt.figure(figsize=(15, 10))
            
            # Plot original audio waveform
            plt.subplot(3, 1, 1)
            plt.plot(np.linspace(0, len(audio)/sr, len(audio)), audio)
            plt.title("Audio Waveform")
            plt.xlabel("Time (s)")
            plt.ylabel("Amplitude")
            
            # Plot the mel spectrogram
            plt.subplot(3, 1, 2)
            librosa.display.specshow(
                mel_db, 
                sr=sr, 
                hop_length=hop_length, 
                x_axis='time', 
                y_axis='mel'
            )
            plt.colorbar(format='%+2.0f dB')
            plt.title(f"Mel Spectrogram with {len(peaks)} Strongest Peaks")
            
            # Plot the peaks
            non_zero_coords = [i for i, p in enumerate(peaks) if not (p[0] == 0 and p[1] == 0 and p[2] == 0)]
            if non_zero_coords:
                actual_coords = final_coordinates[non_zero_coords]
                actual_times = times_seconds[non_zero_coords]
                for coord, time_sec in zip(actual_coords, actual_times):
                    plt.scatter(time_sec, coord[0], color='white', s=10)
            
            # Show original dB values vs normalized values
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.scatter(range(len(final_peak_values)), final_peak_values, alpha=0.7)
            plt.axhline(y=DB_MIN, color='r', linestyle='--', alpha=0.5)
            plt.axhline(y=DB_MAX, color='r', linestyle='--', alpha=0.5)
            plt.title("Original dB Values of Peaks")
            plt.ylabel("Amplitude (dB)")
            plt.xlabel("Peak Index")
            
            plt.subplot(1, 2, 2)
            plt.scatter(range(len(normalized_amplitudes)), normalized_amplitudes, alpha=0.7)
            plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
            plt.axhline(y=1, color='r', linestyle='--', alpha=0.5)
            plt.title("Normalized Amplitude Values (0-1)")
            plt.ylabel("Normalized Amplitude")
            plt.xlabel("Peak Index")
            plt.tight_layout()
            plt.show()
            
            # Plot peaks with normalized amplitudes
            plt.figure(figsize=(15, 5))
            plt.scatter(
                [p[1] for p in peaks],  # Time in seconds
                [p[0] for p in peaks],  # Normalized frequency
                c=[p[2] for p in peaks],  # Normalized amplitude
                cmap='plasma',
                s=30,
                alpha=0.7,
                vmin=0,
                vmax=1
            )
            plt.colorbar(label='Normalized Amplitude')
            plt.title("Extracted Peaks with Normalized Amplitudes")
            plt.xlabel("Time (s)")
            plt.ylabel("Normalized Frequency")
            
            plt.tight_layout()
            plt.show()
        
        return peaks


    def encode_peaks(self, peaks):
          """
            
          
          Args:
              peaks: List of (frequency, time, amplitude) tuples
              
          Returns:
              bytes: Binary encoded data with peak information
          """
          # Validate input
          if not peaks:
              return b''
          
          # Create a bitarray for packing
          bits = bitarray()
          
          # Add a version byte for future compatibility
          version = 3  # Version 3 uses two-part time encoding with 6+12 bits
          bits.frombytes(struct.pack('B', version))
          
          # Add number of peaks (2 bytes, up to 65535 peaks)
          bits.frombytes(struct.pack('<H', len(peaks)))
          
          # Add each peak using bit packing
          for freq, time_seconds, amp in peaks:
              # Convert to integers with specified precision
              freq_int = min(int(round(freq * self.FREQ_SCALE)), self.FREQ_MAX)
              
              # Convert time to minutes and seconds
              minutes = int(time_seconds / 60)
              seconds = time_seconds % 60
              
              # Ensure within range
              minutes_int = min(minutes, self.MINUTE_MAX)
              seconds_int = min(int(round(seconds * self.SECOND_SCALE)), self.SECOND_MAX)
              
              amp_int = min(int(round(amp * self.AMP_SCALE)), self.AMP_MAX)
              
              # Add frequency bits (10 bits)
              for i in range(self.FREQ_BITS - 1, -1, -1):
                  bits.append((freq_int >> i) & 1)
                  
              # Add minutes bits (6 bits)
              for i in range(self.MINUTE_BITS - 1, -1, -1):
                  bits.append((minutes_int >> i) & 1)
                  
              # Add seconds bits (12 bits)
              for i in range(self.SECOND_BITS - 1, -1, -1):
                  bits.append((seconds_int >> i) & 1)
                  
              # Add amplitude bits (10 bits)
              for i in range(self.AMP_BITS - 1, -1, -1):
                  bits.append((amp_int >> i) & 1)
          
          # Convert to bytes
          # Pad to make byte-aligned if necessary
          padding = (8 - (len(bits) % 8)) % 8
          for _ in range(padding):
              bits.append(0)
          
          return bits.tobytes()
    
    def decode_peaks(self, binary_data):
        """
        Decode binary data back to a list of peaks.
        
        Args:
            binary_data: Bytes containing encoded peak data
            
        Returns:
            list: List of (frequency, time, amplitude) tuples
        """
        if not binary_data:
            return []
        
        # Convert to bitarray
        bits = bitarray()
        bits.frombytes(binary_data)
        
        # Extract version (1 byte)
        version_bits = bits[0:8]
        version_bytes = version_bits.tobytes()
        version = struct.unpack('B', version_bytes)[0]
        
        if version != 3:
            raise ValueError(f"Unsupported version: {version}. This decoder requires version 3 (two-part time encoding with 6+12 bits)")
        
        # Extract number of peaks (2 bytes)
        num_peaks_bits = bits[8:24]
        num_peaks_bytes = num_peaks_bits.tobytes()
        num_peaks = struct.unpack('<H', num_peaks_bytes)[0]
        
        # Extract peaks
        peaks = []
        bit_pos = 24  # Start position after header
        
        for _ in range(num_peaks):
            # Extract frequency (10 bits)
            freq_int = 0
            for i in range(self.FREQ_BITS):
                freq_int = (freq_int << 1) | bits[bit_pos + i]
            
            # Extract minutes (6 bits)
            minutes_int = 0
            for i in range(self.FREQ_BITS, self.FREQ_BITS + self.MINUTE_BITS):
                minutes_int = (minutes_int << 1) | bits[bit_pos + i]
                
            # Extract seconds (12 bits)
            seconds_int = 0
            offset = self.FREQ_BITS + self.MINUTE_BITS
            for i in range(offset, offset + self.SECOND_BITS):
                seconds_int = (seconds_int << 1) | bits[bit_pos + i]
                
            # Extract amplitude (10 bits)
            amp_int = 0
            offset = self.FREQ_BITS + self.MINUTE_BITS + self.SECOND_BITS
            for i in range(offset, offset + self.AMP_BITS):
                amp_int = (amp_int << 1) | bits[bit_pos + i]
            
            # Convert back to float values
            freq = freq_int / self.FREQ_SCALE
            
            # Convert minutes and seconds back to total seconds
            seconds = seconds_int / self.SECOND_SCALE
            time_seconds = minutes_int * 60 + seconds
            
            amp = amp_int / self.AMP_SCALE
            
            peaks.append((freq, time_seconds, amp))
            bit_pos += self.BITS_PER_PEAK
        
        return peaks

    def compare_peak(self, binary_data1, binary_data2, time_tolerance=0.05, freq_tolerance=0.01, amp_tolerance=0.05):
        """
        Compare two binary peak data using a sliding window approach.
        First matches corresponding peaks using only time and frequency,
        then calculates similarity using all three dimensions (time, freq, amp).
        """
        # Decode both binary data to get peaks
        peaks1 = self.decode_peaks(binary_data1)
        peaks2 = self.decode_peaks(binary_data2)
        
        # Handle edge cases
        if not peaks1 and not peaks2:
            return 1.0
        if not peaks1 or not peaks2:
            return 0.0
        if len(peaks1) != len(peaks2):
            return 0.0
        
        # Sort both peak lists by time
        peaks1.sort(key=lambda x: x[1])
        peaks2.sort(key=lambda x: x[1])
        
        total_similarity = 0.0
        used_indices = set()
        matched_pairs = []
        window_start = 0
        
        # For each peak in peaks1, find corresponding peak in peaks2
        for i, p1 in enumerate(peaks1):
            min_match_distance = float('inf')
            best_match_idx = None
            
            # Update window_start to skip peaks that are too far behind in time
            while window_start < len(peaks2) and peaks2[window_start][1] < p1[1] - time_tolerance:
                window_start += 1
            
            # Search within window (only peaks within time_tolerance)
            window_end = window_start
            while window_end < len(peaks2) and peaks2[window_end][1] <= p1[1] + time_tolerance:
                if window_end in used_indices:
                    window_end += 1
                    continue
                
                p2 = peaks2[window_end]
                
                # Calculate time and frequency differences for matching
                time_diff = abs(p1[1] - p2[1]) / time_tolerance
                freq_diff = abs(p1[0] - p2[0]) / freq_tolerance
                
                # Use time and frequency only for finding corresponding peaks
                match_distance = (time_diff ** 2 + freq_diff ** 2) ** 0.5
                
                if match_distance < min_match_distance:
                    min_match_distance = match_distance
                    best_match_idx = window_end
                
                window_end += 1
            
            # If found a corresponding peak
            if best_match_idx is not None and min_match_distance < 2.0:  # Threshold for accepting a match
                used_indices.add(best_match_idx)
                p2 = peaks2[best_match_idx]
                matched_pairs.append((p1, p2))
                
                # Now calculate full distance using all three dimensions
                full_distance = (
                    abs(p1[1] - p2[1])  +  # time difference
                    abs(p1[0] - p2[0])  +  # frequency difference
                    abs(p1[2] - p2[2])      # amplitude difference
                )
                print(p1, p2 ,full_distance)
                # Add to total similarity (1.0 for perfect match, 0.0 for maximum difference)
                similarity = max(0.0, 1.0 - (full_distance / 3.0))  # Divide by 3 since we sum 3 differences
                total_similarity += similarity
        
        print("Matched points:", len(matched_pairs))
            
        return total_similarity / len(peaks1)
    
    def compare_visually(self, binary_data1, binary_data2):
        """
        Visually compare two binary peak data sets by plotting them.
        
        Args:
            binary_data1: First binary peak data
            binary_data2: Second binary peak data
            
        Returns:
            None: Creates and displays/saves two plots:
                 1. Frequency vs Time (ignoring amplitude)
                 2. Amplitude vs Time (ignoring frequency)
        """
        try:
            import matplotlib.pyplot as plt
            
            # Decode both binary data to get peaks
            peaks1 = self.decode_peaks(binary_data1)
            peaks2 = self.decode_peaks(binary_data2)
            
            # Handle edge cases
            if not peaks1 and not peaks2:
                print("Both datasets are empty. Nothing to visualize.")
                return
            
            if not peaks1:
                print("First dataset is empty. Nothing to visualize.")
                return
                
            if not peaks2:
                print("Second dataset is empty. Nothing to visualize.")
                return
            
            # Extract data points
            freq1 = [p[0] for p in peaks1]
            time1 = [p[1] for p in peaks1]
            amp1 = [p[2] for p in peaks1]
            
            freq2 = [p[0] for p in peaks2]
            time2 = [p[1] for p in peaks2]
            amp2 = [p[2] for p in peaks2]
            
            # Create figure with two subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            # Plot 1: Frequency vs Time (ignoring amplitude)
            ax1.scatter(time1, freq1, c='blue', alpha=0.7, label='Dataset 1', s=15)
            ax1.scatter(time2, freq2, c='red', alpha=0.7, label='Dataset 2', s=15)
            ax1.set_title("Frequency vs Time Comparison")
            ax1.set_xlabel("Time (seconds)")
            ax1.set_ylabel("Frequency")
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Amplitude vs Time (ignoring frequency)
            ax2.scatter(time1, amp1, c='blue', alpha=0.7, label='Dataset 1', s=15)
            ax2.scatter(time2, amp2, c='red', alpha=0.7, label='Dataset 2', s=15)
            ax2.set_title("Amplitude vs Time Comparison")
            ax2.set_xlabel("Time (seconds)")
            ax2.set_ylabel("Amplitude")
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save the figure
            output_file = 'peak_comparison.png'
            plt.savefig(output_file)
            print(f"Visual comparison saved to {output_file}")
            
            # Try to display the plot (works in interactive environments)
            try:
                plt.show()
            except:
                pass
                
        except ImportError:
            print("Matplotlib not available. Cannot generate visual comparison.")

    def compare_hungarian(self, binary_data1, binary_data2, time_tolerance=0.01, freq_tolerance=0.01, amp_tolerance=0.05):
        """
        Compare two binary peak data using Hungarian algorithm to find optimal matching
        that minimizes the total Manhattan distance between peaks in time, frequency, and amplitude.
        """
        # Decode both binary data to get peaks
        peaks1 = self.decode_peaks(binary_data1)
        peaks2 = self.decode_peaks(binary_data2)
        
        # Handle edge cases
        if not peaks1 and not peaks2:
            return 1.0
        if not peaks1 or not peaks2:
            return 0.0
        if len(peaks1) != len(peaks2):
            return 0.0
        
        n = len(peaks1)
        # Create cost matrix
        cost_matrix = np.zeros((n, n))
        
        # Fill cost matrix with manhattan distances
        for i, p1 in enumerate(peaks1):
            for j, p2 in enumerate(peaks2):
                # Calculate manhattan distance using all three dimensions
                full_distance = (
                    abs(p1[1] - p2[1])/time_tolerance  +  # time difference
                    abs(p1[0] - p2[0])/freq_tolerance +  # frequency difference
                    abs(p1[2] - p2[2])/amp_tolerance    # amplitude difference
                )
                cost_matrix[i, j] = full_distance
        
        # Use Hungarian algorithm to find optimal matching
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # Calculate similarity from matched pairs
        total_similarity = 0.0
        matched_pairs = []
        
        for i, j in zip(row_ind, col_ind):
            p1 = peaks1[i]
            p2 = peaks2[j]
            distance = cost_matrix[i, j]
            matched_pairs.append((p1, p2))
            
            # Convert distance to similarity score
            similarity = max(0.0, 1.0 - (distance / 3.0))
            total_similarity += similarity
        
        print("Matched points:", len(matched_pairs))
        
        return total_similarity / n


if __name__ == "__main__":
    audio_path = "/Users/kanishka/Downloads/Subtitled Video.mp3"
    start_time = time.time() 
    peaks = AudioHash.create_time_sensitive_hash(audio_path)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Peak generation took {execution_time:.5f} seconds")
    print(peaks)
        