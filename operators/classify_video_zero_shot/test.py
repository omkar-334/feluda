import pytest

from feluda.factory import VideoFactory
from operators.classify_video_zero_shot import VideoClassifier


@pytest.fixture(scope="module")
def operator():
    """Fixture to provide a VideoClassifier instance for tests."""
    return VideoClassifier()


def test_sample_video_from_url(operator: VideoClassifier):
    """Test video classification with a sample video from URL."""
    video_url = (
        "https://tattle-media.s3.amazonaws.com/test-data/tattle-search/cat_vid_2mb.mp4"
    )
    file = VideoFactory.make_from_url(video_url)
    labels = ["cat", "dog"]
    result = operator.run(file, labels)

    assert result["prediction"] in labels
    assert isinstance(result["probs"], list)
    assert len(result["probs"]) == len(labels)


@pytest.mark.skip(reason="This test requires a local video file.")
def test_sample_video_from_disk(operator: VideoClassifier):
    """Test video classification with a local video file."""
    file = VideoFactory.make_from_file_on_disk(
        "core/operators/sample_data/sample-cat-video.mp4"
    )
    labels = ["cat", "dog"]
    result = operator.run(file, labels)

    assert result["prediction"] in labels
    assert isinstance(result["probs"], list)
    assert len(result["probs"]) == len(labels)


def test_initialization_ffmpeg_not_found(monkeypatch):
    """Test that initialization fails when FFmpeg is not available."""
    monkeypatch.setattr("shutil.which", lambda _: None)
    with pytest.raises(RuntimeError, match="FFmpeg is not installed"):
        VideoClassifier()


def test_run_invalid_file_object(operator: VideoClassifier):
    """Test that run fails with invalid file object."""
    with pytest.raises(TypeError, match="Invalid file object"):
        operator.run("not_a_dict", ["cat", "dog"])


def test_run_file_not_found(operator: VideoClassifier):
    """Test that run fails when file does not exist."""
    with pytest.raises(FileNotFoundError):
        operator.run({"path": "fake.mp4"}, ["cat", "dog"])


def test_run_empty_labels(operator: VideoClassifier):
    """Test that run fails with empty labels list."""
    video_url = (
        "https://tattle-media.s3.amazonaws.com/test-data/tattle-search/cat_vid_2mb.mp4"
    )
    file = VideoFactory.make_from_url(video_url)
    with pytest.raises(ValueError, match="Label list must not be empty"):
        operator.run(file, [])


def test_run_labels_not_list(operator: VideoClassifier):
    """Test that run fails when labels is not a list."""
    video_url = (
        "https://tattle-media.s3.amazonaws.com/test-data/tattle-search/cat_vid_2mb.mp4"
    )
    file = VideoFactory.make_from_url(video_url)
    with pytest.raises(TypeError, match="labels must be a list of strings"):
        operator.run(file, "cat")


def test_run_labels_not_str(operator: VideoClassifier):
    """Test that run fails when labels contains non-string elements."""
    video_url = (
        "https://tattle-media.s3.amazonaws.com/test-data/tattle-search/cat_vid_2mb.mp4"
    )
    file = VideoFactory.make_from_url(video_url)
    with pytest.raises(TypeError, match="labels must be a list of strings"):
        operator.run(file, [1, 2, 3])
