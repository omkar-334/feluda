# VideoClassifier Operator

## Description

The `VideoClassifier` operator classifies a video into user-provided labels using the CLIP-ViT-B-32 model in a zero-shot fashion. It works by extracting I-frames (keyframes) from a video file using FFmpeg, then using the CLIP model to predict the most likely label for the video content by comparing frame features to text label embeddings.

## Model Information

- **Model**: [CLIP ViT-B/32](https://huggingface.co/openai/clip-vit-base-patch32)
- **Source**: OpenAI, via HuggingFace Transformers
- **Usage**: Zero-shot classification of video content by comparing extracted frame features to text label embeddings.

## System Dependencies

- Python >= 3.10
- FFmpeg
  - On Windows, you have two methods -
      1. Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH.
      2. Use `winget install ffmpeg` from an elevated powershell. (Make sure you have winget installed first)
  - On Linux/macOS, install via your package manager (e.g., `sudo apt install ffmpeg`).

## Operator Dependencies

- PyTorch >= 2.6.0
- Torchvision >= 0.21.0
- Transformers >= 4.51.1
- Pillow >= 11.1.0

## How to Run the Tests

1. Ensure that you are in the root directory of the `feluda` project.
2. Install dependencies (in your virtual environment):

   ```bash
   uv pip install "./operators/classify_video_zero_shot"
   uv pip install "feluda[dev]"
   ```

3. Ensure FFmpeg is installed and available in your PATH.
4. Run the tests:

   ```bash
   pytest operators/classify_video_zero_shot/test.py
   ```

## Usage

```python
from feluda.factory import VideoFactory
from operators.classify_video_zero_shot import VideoClassifier

# Initialize the operator
operator = VideoClassifier()

# Load a video
video_url = (
    "https://tattle-media.s3.amazonaws.com/test-data/tattle-search/cat_vid_2mb.mp4"
)
file = VideoFactory.make_from_url(video_url)

# Classify the video
labels = ["cat", "dog"]
result = operator.run(file, labels)
print(result)

# Cleanup
operator.cleanup()
```

Output:
```json
{"prediction": "cat", "probs": [0.9849101901054382, 0.015089876018464565]}
```
