# *Clipify* - Smart Video Silence Trimmer

## Overview

This Python script intelligently trims silent parts from video files using advanced speech detection and audio analysis techniques. It leverages the Whisper AI model for speech recognition and provides multiple options for refining video editing.

## Features

- 🎙️ Speech Detection: Uses OpenAI's Whisper model to accurately detect speech segments
- 🔇 Silence Trimming: Removes unnecessary pauses and silent periods from videos
- 🧹 Advanced Cleaning Options:
  - Remove filler words (um, uh, like, etc.)
  - Detect and remove mid-sentence pauses
  - Identify and trim repetitive phrases
  - Audio-based filler sound detection
- 🎨 Preserves Original Video Quality:
  - Maintains original resolution
  - Keeps video rotation metadata
  - Supports various video codecs

## Prerequisites

- Python 3.8+
- FFmpeg installed on your system

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/smart-video-trimmer.git
   cd smart-video-trimmer
   ```

2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Basic Usage
```bash
python video_silence_trimmer.py input_video.mp4
```

### Advanced Options
```bash
# Create a default configuration file
python video_silence_trimmer.py --create-config

# Trim video with custom configuration
python video_silence_trimmer.py input_video.mp4 --config my_config.yaml

# Remove filler words and detect mid-sentence pauses
python video_silence_trimmer.py input_video.mp4 --remove-fillers --detect-mid-sentence-pauses
```

### Command-Line Options
- `video_file`: Input video file (required)
- `--output, -o`: Specify output video file
- `--config, -c`: Use a custom configuration file
- `--create-config`: Generate a default configuration file
- `--debug`: Enable detailed debug output
- `--save-transcript`: Save speech recognition transcript
- `--remove-fillers`: Remove filler words
- `--detect-rephrasing`: Remove repeated phrases
- `--detect-word-repetitions`: Remove repetitive word sequences
- `--detect-mid-sentence-pauses`: Remove long pauses within sentences
- `--audio-filler-detection`: Detect audio-based filler sounds

## Configuration

The script uses a YAML configuration file with extensive customization options:
- Silence detection thresholds
- Speech recognition model selection
- Filler word removal
- Pause detection sensitivity
- Video encoding parameters

Generate a default configuration with:
```bash
python video_silence_trimmer.py --create-config
```

## Dependencies

- MoviePy: Video processing
- OpenAI Whisper: Speech recognition
- NumPy: Numerical computing
- PyYAML: Configuration file handling

## Performance Considerations

- Larger Whisper models (medium, large) provide better accuracy but are slower
- Set appropriate thresholds in the configuration for best results
- Recommended minimum system requirements:
  - 8GB RAM
  - Multi-core CPU
  - SSD for faster processing

## License

[Specify your license here, e.g., MIT License]

## Contributing

Contributions are welcome! Please submit pull requests or open issues on the GitHub repository.

## Troubleshooting

- Ensure FFmpeg is installed and accessible in your system PATH
- Check that input video has an audio track
- Adjust configuration parameters if results are not satisfactory

## Future Improvements

- Support for multiple languages
- Machine learning-based speech quality detection
- More granular editing options
