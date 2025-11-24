# StimImager

StimImager is a Python-based audio generation tool for creating "stimphonies".
Inspired by fallen_angel's stimphony creation web app due to unfortunately not being available reliably at the moment.

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
In the [examples/](examples/) directory you can find some example patterns that you can use as a starting point.

## Output

Generated audio files are saved in the [stimfiles/](stimfiles/) directory by default. This directory is included in `.gitignore` to prevent accidental commits of generated files.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
Please read [CONTRIBUTING](CONTRIBUTING.md) before.