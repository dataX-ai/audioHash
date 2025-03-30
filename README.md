# Audio Peak Detection and Fingerprinting

A sophisticated audio analysis tool that creates time-sensitive fingerprints from audio files using mel spectrogram peak detection.

## 📝 Detailed Documentation

For complete project documentation and technical details, visit our [comprehensive report](https://dot-diplodocus-01b.notion.site/Voice-Studio-1c6b48db696080a9b671dab9a3eb8f51).

## 🎯 Features

- Advanced mel spectrogram analysis
- Adaptive peak detection algorithm
- Time-sensitive audio fingerprinting
- Robust audio comparison capabilities
- Visualization tools for analysis

## 🔍 Technical Overview

This project implements an advanced audio fingerprinting system that:

- Converts audio to mel spectrograms using librosa
- Employs adaptive threshold techniques for peak detection
- Normalizes frequency, time, and amplitude data
- Generates consistent fingerprints for audio comparison

### Key Components

- **Mel Spectrogram Generation**: Uses 128 mel bands with 2048 FFT window size
- **Peak Detection**: Adaptive algorithm with configurable parameters
- **Normalization**: Frequency (0-1), Time (seconds), Amplitude (0-1 dB-scaled)

## 🚀 Getting Started

### Prerequisites

- Python 3.x
- librosa
- numpy
- scipy
- matplotlib (for visualization)

### Installation

```bash
pip install -r requirements.txt
```

### Basic Usage

```python
from audioHash import AudioHasher

hasher = AudioHasher()
peaks = hasher.create_time_sensitive_hash('path/to/audio.wav', num_peaks=256)
```

## 📊 Visualization

The system includes comprehensive visualization tools:

- Waveform display
- Mel spectrogram with peak overlay
- Peak amplitude distribution
- Normalized peak visualization

## 🔧 Parameters

- `num_peaks`: Number of peaks to detect (default: 256)
- `n_mels`: Number of mel bands (default: 128)
- `hop_length`: Spectrogram hop length (default: 512)
- `n_fft`: FFT window size (default: 2048)

## 📈 Performance

- Adaptive threshold ensures consistent peak detection
- Efficient processing for various audio lengths
- Robust to volume variations
- Time-sensitive matching capabilities

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

MIT License

Copyright (c) 2024 [Your Name or Organization]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
