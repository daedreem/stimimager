# StimImager

StimImager is a Python-based audio composition tool that generates dynamic audio patterns using various sound effects and waveforms. It's designed to create complex audio stimulations with customizable parameters.

## Features

- Generate audio compositions with multiple waveform types
- Apply various audio effects including:
  - Tremolo (amplitude modulation)
  - Vibrato (frequency modulation)
  - Phaser
  - Distortion
  - Chorus
  - Stutter effects
  - Filter sweeps
  - Bitcrushing
- Customizable frequency ranges and amplitude controls
- Multi-layered audio generation
- Crossfading between segments for smooth transitions
- Export to MP3 format

## Installation
More about how to install can be found in the [DOCUMENTATION](DOCUMENTATION.md).

## Usage

Basic usage:
```bash
   python main.py --freq-range 500 1500 TTGHPO --output basic.mp3
```

## Examples
In the `examples/` directory you can find some example patterns that you can use as a starting point.

## Output

Generated audio files are saved in the `stimfiles/` directory by default. This directory is included in `.gitignore` to prevent accidental commits of generated files.

## License

This project is open source and available under the [MIT License](LICENSE).

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgements

- Built with NumPy for numerical operations
- Audio processing with SoundFile and PyDub
- FFmpeg for MP3 encoding
