# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python-based WAV audio file analyzer that provides comprehensive visualization and analysis of audio waveforms, frequency spectrums, and EQ characteristics. The tool is designed for music production, audio analysis, and educational purposes.

## Core Architecture

The project consists of a single main Python script `visualize_wav.py` that implements multiple analysis functions:

- **Waveform visualization**: Basic time-domain audio visualization
- **Frequency spectrum analysis**: EQ characteristics with musical frequency band mapping
- **Spectrogram generation**: Time-frequency representation
- **Octave band analysis**: ISO 266 compliant octave band measurements
- **Time-evolving EQ analysis**: Heatmap and 3D visualization of frequency changes over time
- **Instant EQ analysis**: Detailed analysis at specific time points
- **Emphasis analysis**: Musical interpretation of frequency band emphasis

## Key Components

### Musical Frequency Bands
The system uses 8 predefined frequency bands with musical significance:
- Low Air (20-100Hz): Bass extension and spatial depth
- Low Thickness (100-140Hz): Kick drum and bass power
- Low Stability (220-280Hz): Musical foundation
- Warmth (350-450Hz): Instrument warmth and body
- Mid Body (800-1000Hz): Vocal and instrument presence
- Brightness (1800-2200Hz): Clarity and forward projection
- Edge (3500-4500Hz): Attack and definition
- High Air (8000Hz+): High-frequency sparkle and space

### Analysis Functions
- `plot_waveform()`: Basic waveform visualization
- `plot_frequency_spectrum()`: EQ analysis with band coloring
- `plot_emphasis_analysis()`: Statistical emphasis analysis
- `plot_instant_eq()`: Detailed time-point analysis
- `plot_eq_evolution()`: Time-based EQ change heatmap
- `plot_time_frequency_analysis()`: 3D time-frequency analysis
- `plot_octave_analysis()`: ISO standard octave band analysis
- `plot_spectrogram()`: Traditional spectrogram

## Common Commands

### Installation
```bash
pip install -r requirements_viz.txt
```

### Basic Usage
```bash
# Basic waveform visualization
python visualize_wav.py input/audio.wav

# EQ analysis
python visualize_wav.py input/audio.wav --eq

# Emphasis analysis
python visualize_wav.py input/audio.wav --emphasis

# Instant EQ at specific time
python visualize_wav.py input/audio.wav --instant "1:30"

# All analyses
python visualize_wav.py input/audio.wav --all
```

### Testing
No formal test suite is present. Manual testing involves running the tool with sample WAV files.

## Dependencies

The project requires three main Python packages:
- `matplotlib>=3.5.0`: For all visualization
- `librosa>=0.9.0`: For audio processing and analysis
- `numpy>=1.21.0`: For numerical computations

## File Structure

- `visualize_wav.py`: Main script containing all analysis functions
- `requirements_viz.txt`: Python dependencies
- `VISUALIZE_WAV_USAGE.md`: Comprehensive Japanese documentation with usage examples
- Input audio files should be placed in an `input/` directory

## Development Notes

### Code Style
- Functions are well-documented with docstrings
- Uses matplotlib for all plotting with consistent styling
- Implements proper error handling for file operations and time parsing
- Japanese comments and output messages (internationalization consideration needed)

### Key Technical Details
- Audio loaded using librosa with configurable sample rates
- FFT analysis with positive frequency filtering
- dB conversion with epsilon addition to prevent log(0)
- Window-based analysis for time-evolving features
- Musical frequency band mapping with color coding
- Statistical emphasis analysis using mean and standard deviation

### Extension Points
- Additional frequency band definitions
- Different window functions for FFT analysis
- Export formats beyond PNG
- Real-time analysis capabilities
- Batch processing for multiple files