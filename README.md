# DAVA: Data Audio/Video Analysis Pipeline

A comprehensive pipeline for processing audio and video files with Automatic Speech Recognition (ASR), speaker diarization, forced alignment, and topic detection.

## Features

- **Automatic Speech Recognition (ASR)**: Supports multiple backends (Transformers, OpenAI Whisper)
- **Speaker Diarization**: Identifies and separates different speakers in audio
- **Forced Alignment**: Word-level timestamp alignment for accurate transcription
- **Language Detection**: Automatic language detection at file and segment levels
- **Topic Classification**: AI-powered topic detection using Qwen models
- **Batch Processing**: Efficient GPU-based batch processing for large datasets
- **Flexible Input**: Supports multiple audio/video formats (MP4, MP3, WAV, FLAC, etc.)

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Usage](#usage)
- [Pipeline Stages](#pipeline-stages)
- [Output Format](#output-format)
- [Requirements](#requirements)
- [Troubleshooting](#troubleshooting)

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- Linux/macOS (Windows support may vary)

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd DAVA
```

### Step 2: Set Up Main Python Environment

```bash
# Create virtual environment
python3 -m venv dava-env
source dava-env/bin/activate  # On Windows: dava-env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install pyyaml  # Required for config file parsing
```

### Step 3: Set Up Qwen Environment (for Topic Detection)

If you plan to use topic detection, set up a separate environment for Qwen:

```bash
# Create separate environment for Qwen
python3 -m venv qwen3-env
source qwen3-env/bin/activate

# Install Qwen dependencies
pip install torch transformers
# Install Qwen model (will be downloaded on first use)
```

### Step 4: Configure HuggingFace Token

For speaker diarization, you need a HuggingFace token:

1. Get your token from [HuggingFace Settings](https://huggingface.co/settings/tokens)
2. Accept the terms for `pyannote/speaker-diarization-3.1` model
3. Add the token to your `config.yaml` (see Configuration section)

## Quick Start

### 1. Configure Your Settings

Copy the example configuration file:

```bash
cp config.yaml.example config.yaml
```

Edit `config.yaml` with your paths and settings (see [Configuration](#configuration) section).

### 2. Run the Pipeline

The pipeline has two stages:

**Stage 1: Generate Manifest**
```bash
./run_my_data_AG.sh --stage 1 --stop_stage 1
```

This creates a CSV manifest file listing all audio/video files in your data directory.

**Stage 2: Process Files**
```bash
./run_my_data_AG.sh --stage 2 --stop_stage 2
```

This processes all files in the manifest through the ASR pipeline.

**Run Both Stages:**
```bash
./run_my_data_AG.sh --stage 1 --stop_stage 2
```

## Configuration

The pipeline is configured via `config.yaml`. Key settings:

### Data Paths

```yaml
data_dir: "/path/to/your/data"  # Directory containing audio/video files
file_extension: "mp4"  # File type to process (mp4, mp3, wav, flac, etc.)
csv_file: "/path/to/manifest.csv"  # Manifest file path
out_dir: "/path/to/output"  # Output directory
```

### ASR Configuration

```yaml
asr_backend: "transformers"  # "transformers" or "openai"
asr_model: "distil-whisper/distil-large-v3.5"  # Model name
lang: ""  # Language code (empty = auto-detect)
lang_detection_model: "base"  # Model size for language detection
```

### Processing Settings

```yaml
gpu_index: 0  # GPU to use
batch_size: 32  # Files per batch
num_speakers: ""  # Number of speakers (empty = auto-detect)
hf_token: "your_huggingface_token"  # Required for diarization
```

### Topic Detection

```yaml
enable_topic_detection: true
topic_model_id: "Qwen/Qwen3-0.6B"
qwen_python_path: "/path/to/qwen3-env/bin/python3"
topic_verbose: false  # Set to true for debugging
```

### Environment Paths

```yaml
dava_python_path: "/path/to/dava-env/bin/python"
```

## Usage

### Command-Line Options

The main script `run_my_data_AG.sh` accepts:

```bash
./run_my_data_AG.sh [OPTIONS]

Options:
  --config FILE    Configuration YAML file (default: config.yaml)
  --stage N        Start stage (1: manifest, 2: processing)
  --stop_stage N   Stop stage
```

### Processing Single Files

For processing individual files, use:

```bash
python process_single_file_pipeline_AG.py \
    --episode_path /path/to/video.mp4 \
    --out_dir /path/to/output \
    --gpu_index 0 \
    --hf_token your_token
```

### Advanced Options

See help for detailed options:

```bash
python process_batch_gpu.py --help
python process_single_file_pipeline_AG.py --help
python qwen_topic_detector.py --help
```

## Pipeline Stages

### Stage 1: Manifest Generation

- Scans the data directory for files with the specified extension
- Creates a CSV manifest file listing all files to process
- Output: `csv_file` (e.g., `manifest.csv`)

### Stage 2: Batch Processing

For each file in the manifest:

1. **Audio Conversion**: Converts video/audio to WAV format (16kHz, mono)
2. **ASR Transcription**: Transcribes audio using selected ASR model
3. **Language Detection**: Detects language at file and segment levels
4. **Forced Alignment**: Aligns words with timestamps
5. **Speaker Diarization**: Identifies and labels speakers
6. **Topic Detection** (optional): Classifies content topics

Output files are saved in:
- `out_dir/wav/`: Converted audio files
- `out_dir/json/`: JSON files with full transcription and metadata

## Output Format

### JSON Structure

Each processed file generates a JSON file with the following structure:

```json
{
  "language": "en",
  "language_probability": 0.99,
  "segments": [
    {
      "start": 0.0,
      "end": 5.2,
      "text": "Hello, this is a sample transcription.",
      "words": [
        {
          "word": "Hello",
          "start": 0.0,
          "end": 0.5,
          "score": 0.95
        }
      ],
      "speaker": "SPEAKER_00",
      "detected_language": "en",
      "detected_language_probability": 0.98,
      "topic": "News"
    }
  ]
}
```

### Field Descriptions

- `language`: Detected language for the entire file
- `language_probability`: Confidence score for language detection
- `segments`: Array of transcribed segments
  - `start`/`end`: Timestamps in seconds
  - `text`: Transcribed text
  - `words`: Word-level timestamps and scores
  - `speaker`: Speaker identifier (SPEAKER_00, SPEAKER_01, etc.)
  - `detected_language`: Language detected for this segment
  - `topic`: Topic category (if topic detection enabled)

## Requirements

### System Requirements

- **GPU**: NVIDIA GPU with CUDA support (recommended)
- **RAM**: 16GB+ recommended
- **Storage**: Sufficient space for output files (typically 2-5x input size)

### Python Dependencies

See `requirements.txt` for full list. Key dependencies:

- PyTorch (with CUDA support)
- Transformers
- WhisperX
- OpenAI Whisper
- PyYAML
- SoundFile
- NumPy

### Model Requirements

- **ASR Models**: Downloaded automatically from HuggingFace
- **Diarization Model**: Requires HuggingFace token and model acceptance
- **Topic Model**: Qwen models (separate environment recommended)

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory

**Solution**: Reduce `batch_size` in `config.yaml` or use a smaller ASR model.

#### 2. HuggingFace Authentication Error

**Solution**: 
- Ensure your `hf_token` is set correctly in `config.yaml`
- Accept model terms at https://huggingface.co/pyannote/speaker-diarization-3.1

#### 3. Topic Detection Not Working

**Solution**:
- Verify `qwen_python_path` points to correct environment
- Check that Qwen environment has required packages installed
- Enable `topic_verbose: true` for debugging

#### 4. Language Detection Issues

**Solution**:
- Set `lang` in config to force a specific language
- Increase `lang_detection_model` size (e.g., "base" → "small")

#### 5. File Not Found Errors

**Solution**:
- Verify all paths in `config.yaml` are absolute and correct
- Ensure data directory exists and contains files with specified extension

### Performance Optimization

1. **GPU Selection**: Use `gpu_index` to select the GPU with most free memory
2. **Batch Size**: Adjust `batch_size` based on GPU memory (start with 32)
3. **Model Selection**: Use smaller models (e.g., "base") for faster processing
4. **Skip Alignment**: Use `--skip_alignment` flag to skip word-level alignment for speed

### Getting Help

- Check logs in terminal output for detailed error messages
- Enable verbose mode: `topic_verbose: true` in config
- Review JSON output files for processing status


## Acknowledgments

- WhisperX for ASR and diarization
- OpenAI Whisper for language detection
- Qwen for topic classification
- PyAnnote for speaker diarization models

