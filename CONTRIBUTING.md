# Contributing

## Table of Contents

- [Installation](#installation)
- [Adding new effects](#adding-new-effects)
- [Adding new Composition Parts](#adding-new-composition-parts)
- [Adding new demo files](#adding-new-demo-files)
    - [Requirements](#requirements)

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


## Adding new effects

- Add your effect to the `AVAILABLE_EFFECTS` dictionary in `main.py`.
- Write a function like `apply_tremolo` to apply your effect in `main.py`.
- Add your effect to the `apply_effects` function in `main.py`.

## Adding new Composition Parts

- Add your Composition Part to the `COMPOSITION_PARTS` list in `main.py`.
- Generate a demo file using the new Composition Part and add it to the [demo/](demo/) directory, as specified in ["Adding new demo files"](#adding-new-demo-files).


## Adding new demo files

- Simply create a new file using the tool.
- Add your demo file to the [demo](demo/) directory.

### Requirements
- The file should be generated with a duration of 300 seconds.
- The exact used command to generate the file should be added to the [demo](demo/) directorys' [README.md](demo/README.md).