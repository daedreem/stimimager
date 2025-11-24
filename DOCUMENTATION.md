# StimImager Documentation

## Table of Contents

1. [Installation](#installation)
   - [Prerequisites](#prerequisites)
   - [Setting Up the Environment](#setting-up-the-environment)
   - [Installing Dependencies](#installing-dependencies)
2. [Basic Usage](#basic-usage)
   - [Command Line Arguments](#command-line-arguments)
   - [Example Commands](#example-commands)
3. [Troubleshooting](#troubleshooting)

## Installation

### Prerequisites

- Python 3.7 or higher
- FFmpeg (for MP3 export)

### Setting Up the Environment

1. **Clone the repository**:

   ```bash
   git clone https://github.com/daedreem/stimimager.git
   cd stimimager
   ```

2. **Create and activate a virtual environment** (recommended):

   ```bash
   # On Windows
   python -m venv venv
   .\venv\Scripts\activate
   ```

   ```bash
   # On macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

### Installing Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

Install FFmpeg:

- **Ubuntu/Debian**: `sudo apt install ffmpeg`
- **macOS**: `brew install ffmpeg`
- **Windows**: Download from [FFmpeg's website](https://ffmpeg.org/download.html)

## Basic Usage

### Command Line Arguments

| Argument        | Description                       | Default      |
| --------------- | --------------------------------- | ------------ |
| `--output`      | Output file name                  | "output.mp3" |
| `--duration`    | Total duration in seconds         | 300          |
| `--freq-range`  | Frequency range in Hz             | 500 1500     |
| `--sample-rate` | Sample rate in Hz                 | 44100        |
| `--pattern`     | Composition pattern (e.g., "TGO") | -            |
| `--crossfade`   | Crossfade duration in seconds     | 1.0          |
| `--layers`      | Number of audio layers            | 1            |

### Example Commands

1. **Basic generation**
   To generate a basic composition with a pattern from teasing until climax with a length of 5 minutes

   ```bash
   python main.py --freq-range 500 1500 TTGHPO 300 --output basic.mp3
   ```

2. **Multi-layered composition**
   To generate a multi-layered composition with a pattern from teasing until climax with a length of 5 minutes and 2 layers

   ```bash
   python main.py --freq-range 500 1500 TTGHPO 300 --layers 2 --output layered.mp3
   ```

## Troubleshooting

1. **FFmpeg not found**

   - Ensure FFmpeg is installed and added to your system PATH
   - On Windows, you might need to restart your terminal after installation

2. **Missing dependencies**

   ```bash
   # If you get import errors, try:
   pip install -r requirements.txt --upgrade
   ```

3. **Audio quality issues**

   - Increase sample rate: `--sample-rate 48000`
   - Try different frequency ranges
   - Experiment with different patterns

4. **Long generation times**
   - Reduce duration or number of layers
   - Lower sample rate (e.g., `--sample-rate 22050`)
