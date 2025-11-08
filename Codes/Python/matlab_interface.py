"""
MATLAB-Python Integration Interface for SWIFT II
Provides interface between Python RL agent and MATLAB PHY layer
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any
import os
import sys


class MATLABInterface:
    """
    Interface for MATLAB PHY layer integration
    
    Provides methods to:
    - Initialize MATLAB engine
    - Run 5G NR channel simulation
    - Get PSD and interference measurements
    - Calculate SINR and throughput
    """
    
    def __init__(self, matlab_path: Optional[str] = None, timeout: float = 300.0):
        """
        Initialize MATLAB interface
        
        Args:
            matlab_path: Path to MATLAB code directory
            timeout: Timeout for MATLAB operations (seconds)
        """
        self.matlab_engine = None
        self.matlab_path = matlab_path
        self.timeout = timeout
        self.is_initialized = False
        
        # Try to import MATLAB engine
        try:
            import matlab.engine
            self.matlab_module = matlab.engine
        except ImportError:
            print("Warning: MATLAB Engine for Python not found.")
            print("To install: python -m pip install matlabengine")
            self.matlab_module = None
    
    def start_engine(self) -> bool:
        """
        Start MATLAB engine
        
        Returns:
            True if successful, False otherwise
        """
        if self.matlab_module is None:
            print("MATLAB Engine module not available")
            return False
        
        try:
            print("Starting MATLAB engine...")
            self.matlab_engine = self.matlab_module.start_matlab()
            
            # Add MATLAB code path
            if self.matlab_path is not None:
                self.matlab_engine.addpath(self.matlab_path, nargout=0)
            
            self.is_initialized = True
            print("MATLAB engine started successfully")
            return True
            
        except Exception as e:
            print(f"Failed to start MATLAB engine: {e}")
            return False
    
    def stop_engine(self):
        """Stop MATLAB engine"""
        if self.matlab_engine is not None:
            try:
                self.matlab_engine.quit()
                print("MATLAB engine stopped")
            except Exception as e:
                print(f"Error stopping MATLAB engine: {e}")
            finally:
                self.matlab_engine = None
                self.is_initialized = False
    
    def simulate_channel(
        self,
        signal_in: np.ndarray,
        channel_params: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate 5G NR channel using MATLAB
        
        Args:
            signal_in: Input signal (complex numpy array)
            channel_params: Optional channel parameters
        
        Returns:
            signal_out: Output signal after channel
            ofdm_response: OFDM channel response
        """
        if not self.is_initialized:
            raise RuntimeError("MATLAB engine not initialized")
        
        try:
            # Convert numpy array to MATLAB format
            signal_in_matlab = self._numpy_to_matlab_complex(signal_in)
            
            # Call MATLAB channel simulation function
            # Assumes my_NRChannel function exists in MATLAB path
            signal_out_matlab, ofdm_response_matlab = self.matlab_engine.my_NRChannel(
                signal_in_matlab,
                'HFS',  # Channel model type
                True,   # Return OFDM response
                nargout=2
            )
            
            # Convert back to numpy
            signal_out = self._matlab_to_numpy_complex(signal_out_matlab)
            ofdm_response = self._matlab_to_numpy_complex(ofdm_response_matlab)
            
            return signal_out, ofdm_response
            
        except Exception as e:
            print(f"MATLAB channel simulation failed: {e}")
            # Return dummy data as fallback
            signal_out = signal_in * (0.8 + 0.1j)  # Simple channel gain
            ofdm_response = np.ones_like(signal_in)
            return signal_out, ofdm_response
    
    def generate_passive_signal(
        self,
        power_dbm: float,
        num_samples: int
    ) -> np.ndarray:
        """
        Generate passive sensor signal using MATLAB
        
        Args:
            power_dbm: Signal power in dBm
            num_samples: Number of samples
        
        Returns:
            Passive signal (complex numpy array)
        """
        if not self.is_initialized:
            raise RuntimeError("MATLAB engine not initialized")
        
        try:
            # Call MATLAB passive signal generation function
            signal_matlab = self.matlab_engine.Passive_Signal(
                float(power_dbm),
                int(num_samples),
                nargout=1
            )
            
            # Convert to numpy
            signal = self._matlab_to_numpy_complex(signal_matlab)
            return signal
            
        except Exception as e:
            print(f"MATLAB passive signal generation failed: {e}")
            # Generate simple Gaussian noise as fallback
            power_watts = 10 ** ((power_dbm - 30) / 10)
            sigma = np.sqrt(power_watts / 2)
            signal = (np.random.randn(num_samples) + 1j * np.random.randn(num_samples)) * sigma
            return signal
    
    def calculate_psd(
        self,
        signal: np.ndarray,
        fs: float,
        num_channels: int = 7
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate Power Spectral Density using MATLAB
        
        Args:
            signal: Input signal
            fs: Sampling frequency
            num_channels: Number of frequency channels
        
        Returns:
            psd_values: PSD values per channel (linear scale)
            frequencies: Frequency bins
        """
        if not self.is_initialized:
            # Use numpy FFT as fallback
            return self._calculate_psd_numpy(signal, fs, num_channels)
        
        try:
            signal_matlab = self._numpy_to_matlab_complex(signal)
            
            # Call MATLAB PSD calculation function
            psd_matlab, freq_matlab = self.matlab_engine.my_pspectrum(
                signal_matlab,
                float(fs),
                'power',
                'Signal Spectrum',
                nargout=2
            )
            
            # Convert to numpy
            psd = self._matlab_to_numpy(psd_matlab)
            frequencies = self._matlab_to_numpy(freq_matlab)
            
            # Bin PSD into channels
            psd_binned = self._bin_psd_into_channels(psd, frequencies, num_channels)
            
            return psd_binned, frequencies
            
        except Exception as e:
            print(f"MATLAB PSD calculation failed: {e}")
            return self._calculate_psd_numpy(signal, fs, num_channels)
    
    def _calculate_psd_numpy(
        self,
        signal: np.ndarray,
        fs: float,
        num_channels: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate PSD using numpy FFT (fallback)
        
        Args:
            signal: Input signal
            fs: Sampling frequency
            num_channels: Number of frequency channels
        
        Returns:
            psd_values: PSD per channel
            frequencies: Frequency bins
        """
        # FFT
        fft_result = np.fft.fftshift(np.fft.fft(signal))
        power_spectrum = np.abs(fft_result) ** 2 / (len(signal) * fs)
        
        # Frequency axis
        frequencies = np.fft.fftshift(np.fft.fftfreq(len(signal), 1/fs))
        
        # Bin into channels
        psd_binned = self._bin_psd_into_channels(power_spectrum, frequencies, num_channels)
        
        return psd_binned, frequencies
    
    def _bin_psd_into_channels(
        self,
        psd: np.ndarray,
        frequencies: np.ndarray,
        num_channels: int
    ) -> np.ndarray:
        """
        Bin PSD values into frequency channels
        
        Args:
            psd: Full PSD array
            frequencies: Frequency array
            num_channels: Number of channels to bin into
        
        Returns:
            Binned PSD values per channel
        """
        # Calculate frequency bin edges
        f_min = frequencies[0]
        f_max = frequencies[-1]
        bin_edges = np.linspace(f_min, f_max, num_channels + 1)
        
        # Bin PSD into channels
        psd_binned = np.zeros(num_channels)
        for i in range(num_channels):
            mask = (frequencies >= bin_edges[i]) & (frequencies < bin_edges[i+1])
            psd_binned[i] = np.mean(psd[mask]) if np.any(mask) else 0.0
        
        return psd_binned
    
    def _numpy_to_matlab_complex(self, array: np.ndarray):
        """Convert numpy complex array to MATLAB format"""
        if self.matlab_module is None:
            return array
        
        # MATLAB expects separate real and imaginary parts
        real_part = np.real(array).flatten().tolist()
        imag_part = np.imag(array).flatten().tolist()
        
        # Create MATLAB complex array
        matlab_array = self.matlab_module.double(
            complex=True,
            size=(len(real_part), 1)
        )
        
        for i in range(len(real_part)):
            matlab_array[i] = complex(real_part[i], imag_part[i])
        
        return matlab_array
    
    def _matlab_to_numpy_complex(self, matlab_array) -> np.ndarray:
        """Convert MATLAB complex array to numpy format"""
        if isinstance(matlab_array, np.ndarray):
            return matlab_array
        
        # Extract real and imaginary parts
        real_part = np.array([x.real for x in matlab_array])
        imag_part = np.array([x.imag for x in matlab_array])
        
        return real_part + 1j * imag_part
    
    def _matlab_to_numpy(self, matlab_array) -> np.ndarray:
        """Convert MATLAB array to numpy format"""
        if isinstance(matlab_array, np.ndarray):
            return matlab_array
        
        return np.array(matlab_array)
    
    def __enter__(self):
        """Context manager entry"""
        self.start_engine()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop_engine()


class SimulatedMATLABInterface:
    """
    Simulated MATLAB interface for testing without MATLAB installation
    
    Provides same API as MATLABInterface but uses Python-based simulation
    """
    
    def __init__(self, **kwargs):
        """Initialize simulated interface"""
        self.is_initialized = True
        print("Using simulated MATLAB interface (no MATLAB engine)")
    
    def start_engine(self) -> bool:
        """Simulated engine start"""
        self.is_initialized = True
        return True
    
    def stop_engine(self):
        """Simulated engine stop"""
        self.is_initialized = False
    
    def simulate_channel(
        self,
        signal_in: np.ndarray,
        channel_params: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate channel with simple fading model
        
        Args:
            signal_in: Input signal
            channel_params: Channel parameters
        
        Returns:
            signal_out: Output signal
            ofdm_response: Channel response
        """
        # Simple Rayleigh fading
        channel_gain = (np.random.randn() + 1j * np.random.randn()) / np.sqrt(2) * 0.8
        signal_out = signal_in * channel_gain
        
        # Add AWGN
        noise_power = 1e-3
        noise = np.sqrt(noise_power/2) * (np.random.randn(len(signal_in)) + 
                                          1j * np.random.randn(len(signal_in)))
        signal_out += noise
        
        ofdm_response = np.ones_like(signal_in) * channel_gain
        
        return signal_out, ofdm_response
    
    def generate_passive_signal(
        self,
        power_dbm: float,
        num_samples: int
    ) -> np.ndarray:
        """
        Generate simulated passive signal
        
        Args:
            power_dbm: Power level in dBm
            num_samples: Number of samples
        
        Returns:
            Passive signal
        """
        power_watts = 10 ** ((power_dbm - 30) / 10)
        sigma = np.sqrt(power_watts / 2)
        signal = (np.random.randn(num_samples) + 1j * np.random.randn(num_samples)) * sigma
        return signal
    
    def calculate_psd(
        self,
        signal: np.ndarray,
        fs: float,
        num_channels: int = 7
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate PSD using FFT
        
        Args:
            signal: Input signal
            fs: Sampling frequency
            num_channels: Number of channels
        
        Returns:
            psd_values: PSD per channel
            frequencies: Frequency bins
        """
        # FFT-based PSD
        fft_result = np.fft.fftshift(np.fft.fft(signal))
        power_spectrum = np.abs(fft_result) ** 2 / (len(signal) * fs)
        frequencies = np.fft.fftshift(np.fft.fftfreq(len(signal), 1/fs))
        
        # Bin into channels
        f_min = frequencies[0]
        f_max = frequencies[-1]
        bin_edges = np.linspace(f_min, f_max, num_channels + 1)
        
        psd_binned = np.zeros(num_channels)
        for i in range(num_channels):
            mask = (frequencies >= bin_edges[i]) & (frequencies < bin_edges[i+1])
            psd_binned[i] = np.mean(power_spectrum[mask]) if np.any(mask) else 0.0
        
        return psd_binned, frequencies
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        pass


def create_matlab_interface(
    use_real_matlab: bool = False,
    matlab_path: Optional[str] = None,
    timeout: float = 300.0
):
    """
    Factory function to create MATLAB interface
    
    Args:
        use_real_matlab: If True, use real MATLAB engine; otherwise simulate
        matlab_path: Path to MATLAB code directory
        timeout: Timeout for operations
    
    Returns:
        MATLAB interface instance
    """
    if use_real_matlab:
        interface = MATLABInterface(matlab_path=matlab_path, timeout=timeout)
        if interface.start_engine():
            return interface
        else:
            print("Falling back to simulated interface")
            return SimulatedMATLABInterface()
    else:
        return SimulatedMATLABInterface()


if __name__ == "__main__":
    # Test MATLAB interface
    print("Testing MATLAB Interface...")
    print("-" * 80)
    
    # Test with simulated interface
    print("\n1. Testing Simulated Interface:")
    with create_matlab_interface(use_real_matlab=False) as matlab:
        # Generate test signal
        num_samples = 1000
        fs = 30.72e6
        test_signal = np.random.randn(num_samples) + 1j * np.random.randn(num_samples)
        
        # Test channel simulation
        signal_out, ofdm_response = matlab.simulate_channel(test_signal)
        print(f"  Channel simulation: Input shape {test_signal.shape}, Output shape {signal_out.shape}")
        
        # Test passive signal generation
        passive_signal = matlab.generate_passive_signal(power_dbm=-120, num_samples=num_samples)
        print(f"  Passive signal: Shape {passive_signal.shape}, Power {10*np.log10(np.mean(np.abs(passive_signal)**2))+30:.2f} dBm")
        
        # Test PSD calculation
        psd_values, frequencies = matlab.calculate_psd(test_signal, fs, num_channels=7)
        print(f"  PSD calculation: {len(psd_values)} channels")
        print(f"  PSD values (dBm): {10*np.log10(psd_values+1e-12)+30}")
    
    print("\n" + "-" * 80)
    print("MATLAB Interface test complete")
