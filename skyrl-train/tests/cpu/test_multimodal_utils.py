"""
Tests for multimodal/VL support utilities in skyrl_train.generators.utils.

Run with:
uv run --isolated --extra dev pytest tests/cpu/test_multimodal_utils.py -v
"""

import base64
import io
import os
import tempfile
from pathlib import Path

import pytest

from skyrl_train.generators.utils import (
    decode_base64_image,
    extract_images_from_conversation,
    get_text_from_multimodal_content,
    is_multimodal_conversation,
    is_multimodal_message,
    load_image_from_path,
)

# Only import PIL if available (tests will skip if not installed)
try:
    from PIL import Image

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


# --- Fixtures ---


@pytest.fixture
def simple_red_image_base64():
    """Create a simple 2x2 red PNG image as base64."""
    if not PIL_AVAILABLE:
        pytest.skip("PIL not available")

    img = Image.new("RGB", (2, 2), color="red")
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    b64 = base64.b64encode(buffer.read()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


@pytest.fixture
def temp_image_file():
    """Create a temporary image file for testing."""
    if not PIL_AVAILABLE:
        pytest.skip("PIL not available")

    img = Image.new("RGB", (4, 4), color="blue")
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        img.save(f, format="PNG")
        temp_path = f.name

    yield temp_path

    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


# --- Tests for is_multimodal_message ---


class TestIsMultimodalMessage:
    """Tests for is_multimodal_message function."""

    def test_text_only_message_string_content(self):
        """Text-only messages with string content should return False."""
        message = {"role": "user", "content": "Hello, how are you?"}
        assert is_multimodal_message(message) is False

    def test_text_only_message_list_content(self):
        """Text-only messages with list content (no images) should return False."""
        message = {
            "role": "user",
            "content": [{"type": "text", "text": "Hello, how are you?"}],
        }
        assert is_multimodal_message(message) is False

    def test_multimodal_message_with_image(self):
        """Messages with image_url content should return True."""
        message = {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc123"}},
            ],
        }
        assert is_multimodal_message(message) is True

    def test_multimodal_message_image_only(self):
        """Messages with only image content should return True."""
        message = {
            "role": "user",
            "content": [{"type": "image_url", "image_url": {"url": "https://example.com/image.png"}}],
        }
        assert is_multimodal_message(message) is True

    def test_multimodal_message_multiple_images(self):
        """Messages with multiple images should return True."""
        message = {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc123"}},
                {"type": "text", "text": "Compare these images"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,def456"}},
            ],
        }
        assert is_multimodal_message(message) is True

    def test_empty_content_list(self):
        """Messages with empty content list should return False."""
        message = {"role": "user", "content": []}
        assert is_multimodal_message(message) is False

    def test_missing_content_key(self):
        """Messages without content key should return False (not crash)."""
        message = {"role": "user"}
        assert is_multimodal_message(message) is False

    def test_none_content(self):
        """Messages with None content should return False."""
        message = {"role": "user", "content": None}
        assert is_multimodal_message(message) is False

    def test_assistant_message_with_image(self):
        """Assistant messages with images should also be detected."""
        message = {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Here's an image I generated:"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,xyz789"}},
            ],
        }
        assert is_multimodal_message(message) is True


# --- Tests for is_multimodal_conversation ---


class TestIsMultimodalConversation:
    """Tests for is_multimodal_conversation function."""

    def test_text_only_conversation(self):
        """Conversations with only text messages should return False."""
        conversation = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi! How can I help?"},
            {"role": "user", "content": "What's 2+2?"},
        ]
        assert is_multimodal_conversation(conversation) is False

    def test_multimodal_conversation_first_message(self):
        """Conversations where first message has image should return True."""
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is this?"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
                ],
            },
            {"role": "assistant", "content": "It's a cat!"},
        ]
        assert is_multimodal_conversation(conversation) is True

    def test_multimodal_conversation_later_message(self):
        """Conversations where a later message has image should return True."""
        conversation = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Now look at this"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}},
                ],
            },
        ]
        assert is_multimodal_conversation(conversation) is True

    def test_empty_conversation(self):
        """Empty conversations should return False."""
        conversation = []
        assert is_multimodal_conversation(conversation) is False

    def test_multiple_images_across_messages(self):
        """Conversations with images in multiple messages should return True."""
        conversation = [
            {
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,img1"}}],
            },
            {"role": "assistant", "content": "I see image 1"},
            {
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,img2"}}],
            },
        ]
        assert is_multimodal_conversation(conversation) is True


# --- Tests for extract_images_from_conversation ---


class TestExtractImagesFromConversation:
    """Tests for extract_images_from_conversation function."""

    def test_text_only_conversation_returns_empty(self):
        """Text-only conversations should return empty list."""
        conversation = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
        ]
        images = extract_images_from_conversation(conversation)
        assert images == []

    def test_text_only_list_content_returns_empty(self):
        """Messages with list content but no images should return empty list."""
        conversation = [
            {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
        ]
        images = extract_images_from_conversation(conversation)
        assert images == []

    @pytest.mark.skipif(not PIL_AVAILABLE, reason="PIL not installed")
    def test_extract_base64_image(self, simple_red_image_base64):
        """Base64 images should be decoded to PIL Images."""
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's this?"},
                    {"type": "image_url", "image_url": {"url": simple_red_image_base64}},
                ],
            },
        ]
        images = extract_images_from_conversation(conversation)

        assert len(images) == 1
        assert isinstance(images[0], Image.Image)
        assert images[0].size == (2, 2)

    def test_extract_url_image(self):
        """HTTP/HTTPS URLs should be returned as-is."""
        url = "https://example.com/image.png"
        conversation = [
            {
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": url}}],
            },
        ]
        images = extract_images_from_conversation(conversation)

        assert len(images) == 1
        assert images[0] == url

    def test_extract_http_url(self):
        """HTTP URLs should also be returned as-is."""
        url = "http://example.com/image.jpg"
        conversation = [
            {
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": url}}],
            },
        ]
        images = extract_images_from_conversation(conversation)

        assert len(images) == 1
        assert images[0] == url

    @pytest.mark.skipif(not PIL_AVAILABLE, reason="PIL not installed")
    def test_extract_file_path_image(self, temp_image_file):
        """File paths should be loaded as PIL Images."""
        conversation = [
            {
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": temp_image_file}}],
            },
        ]
        images = extract_images_from_conversation(conversation)

        assert len(images) == 1
        assert isinstance(images[0], Image.Image)
        assert images[0].size == (4, 4)

    @pytest.mark.skipif(not PIL_AVAILABLE, reason="PIL not installed")
    def test_extract_multiple_images_same_message(self, simple_red_image_base64):
        """Multiple images in same message should all be extracted in order."""
        url = "https://example.com/other.png"
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": simple_red_image_base64}},
                    {"type": "text", "text": "vs"},
                    {"type": "image_url", "image_url": {"url": url}},
                ],
            },
        ]
        images = extract_images_from_conversation(conversation)

        assert len(images) == 2
        assert isinstance(images[0], Image.Image)  # Base64 decoded
        assert images[1] == url  # URL passed through

    @pytest.mark.skipif(not PIL_AVAILABLE, reason="PIL not installed")
    def test_extract_images_across_messages(self, simple_red_image_base64):
        """Images across multiple messages should be extracted in order."""
        url = "https://example.com/second.png"
        conversation = [
            {
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": simple_red_image_base64}}],
            },
            {"role": "assistant", "content": "I see image 1"},
            {
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": url}}],
            },
        ]
        images = extract_images_from_conversation(conversation)

        assert len(images) == 2
        assert isinstance(images[0], Image.Image)
        assert images[1] == url

    def test_empty_url_ignored(self):
        """Empty URLs should not add anything to the list."""
        conversation = [
            {
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": ""}}],
            },
        ]
        images = extract_images_from_conversation(conversation)
        assert images == []

    def test_missing_image_url_key(self):
        """Messages with malformed image_url content should be handled gracefully."""
        conversation = [
            {
                "role": "user",
                "content": [{"type": "image_url"}],  # Missing image_url key
            },
        ]
        images = extract_images_from_conversation(conversation)
        assert images == []


# --- Tests for decode_base64_image ---


@pytest.mark.skipif(not PIL_AVAILABLE, reason="PIL not installed")
class TestDecodeBase64Image:
    """Tests for decode_base64_image function."""

    def test_decode_png_with_data_url_prefix(self, simple_red_image_base64):
        """Should decode PNG with full data URL prefix."""
        image = decode_base64_image(simple_red_image_base64)

        assert isinstance(image, Image.Image)
        assert image.size == (2, 2)
        # Check it's actually red
        assert image.getpixel((0, 0))[:3] == (255, 0, 0)

    def test_decode_without_prefix(self):
        """Should decode base64 without data URL prefix."""
        # Create a simple image
        img = Image.new("RGB", (3, 3), color="green")
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        buffer.seek(0)
        b64 = base64.b64encode(buffer.read()).decode("utf-8")

        image = decode_base64_image(b64)

        assert isinstance(image, Image.Image)
        assert image.size == (3, 3)

    def test_decode_jpeg_data_url(self):
        """Should decode JPEG images."""
        img = Image.new("RGB", (5, 5), color="yellow")
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        buffer.seek(0)
        b64 = base64.b64encode(buffer.read()).decode("utf-8")
        data_url = f"data:image/jpeg;base64,{b64}"

        image = decode_base64_image(data_url)

        assert isinstance(image, Image.Image)
        assert image.size == (5, 5)

    def test_invalid_base64_raises_error(self):
        """Invalid base64 should raise an error."""
        with pytest.raises(Exception):  # Could be binascii.Error or other
            decode_base64_image("data:image/png;base64,not_valid_base64!!!")


# --- Tests for load_image_from_path ---


@pytest.mark.skipif(not PIL_AVAILABLE, reason="PIL not installed")
class TestLoadImageFromPath:
    """Tests for load_image_from_path function."""

    def test_load_existing_image(self, temp_image_file):
        """Should load an existing image file."""
        image = load_image_from_path(temp_image_file)

        assert isinstance(image, Image.Image)
        assert image.size == (4, 4)

    def test_load_nonexistent_file_raises_error(self):
        """Loading a non-existent file should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_image_from_path("/nonexistent/path/to/image.png")


# --- Tests for get_text_from_multimodal_content ---


class TestGetTextFromMultimodalContent:
    """Tests for get_text_from_multimodal_content function."""

    def test_string_content_returned_as_is(self):
        """String content should be returned unchanged."""
        content = "Hello, world!"
        result = get_text_from_multimodal_content(content)
        assert result == "Hello, world!"

    def test_list_content_extracts_text(self):
        """Text items from list content should be extracted and joined."""
        content = [
            {"type": "text", "text": "Hello"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
            {"type": "text", "text": "World"},
        ]
        result = get_text_from_multimodal_content(content)
        assert result == "Hello World"

    def test_list_content_images_only(self):
        """List with only images should return empty string."""
        content = [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
            {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}},
        ]
        result = get_text_from_multimodal_content(content)
        assert result == ""

    def test_list_content_single_text_item(self):
        """Single text item in list should be extracted."""
        content = [{"type": "text", "text": "Just text"}]
        result = get_text_from_multimodal_content(content)
        assert result == "Just text"

    def test_empty_list_returns_empty_string(self):
        """Empty list should return empty string."""
        content = []
        result = get_text_from_multimodal_content(content)
        assert result == ""

    def test_empty_text_items(self):
        """Empty text items should contribute empty strings."""
        content = [
            {"type": "text", "text": ""},
            {"type": "text", "text": "Hello"},
            {"type": "text", "text": ""},
        ]
        result = get_text_from_multimodal_content(content)
        assert result == " Hello "

    def test_missing_text_key(self):
        """Text items without 'text' key should contribute empty string."""
        content = [{"type": "text"}, {"type": "text", "text": "Valid"}]
        result = get_text_from_multimodal_content(content)
        assert result == " Valid"

    def test_none_content_returns_empty_string(self):
        """None content should return empty string."""
        result = get_text_from_multimodal_content(None)
        assert result == ""

    def test_unexpected_content_type_returns_empty(self):
        """Unexpected content types should return empty string."""
        result = get_text_from_multimodal_content(123)
        assert result == ""
        result = get_text_from_multimodal_content({"key": "value"})
        assert result == ""


# --- Tests for InferenceEngineInput multi_modal_data field ---


class TestInferenceEngineInputMultiModal:
    """Tests for multi_modal_data field in InferenceEngineInput."""

    def test_multimodal_data_type_definition(self):
        """InferenceEngineInput should accept multi_modal_data field."""
        from skyrl_train.inference_engines.base import InferenceEngineInput

        # Create a valid InferenceEngineInput with multi_modal_data
        engine_input: InferenceEngineInput = {
            "prompts": None,
            "prompt_token_ids": [[1, 2, 3]],
            "sampling_params": {"temperature": 0.7},
            "session_ids": ["session1"],
            "multi_modal_data": [{"image": ["mock_image_object"]}],
        }

        assert engine_input["multi_modal_data"] is not None
        assert engine_input["multi_modal_data"][0]["image"] == ["mock_image_object"]

    def test_multimodal_data_can_be_none(self):
        """multi_modal_data should be optional (None for text-only)."""
        from skyrl_train.inference_engines.base import InferenceEngineInput

        engine_input: InferenceEngineInput = {
            "prompts": None,
            "prompt_token_ids": [[1, 2, 3]],
            "sampling_params": None,
            "session_ids": None,
            "multi_modal_data": None,
        }

        assert engine_input["multi_modal_data"] is None

    def test_multimodal_data_per_prompt(self):
        """multi_modal_data should support per-prompt image lists."""
        from skyrl_train.inference_engines.base import InferenceEngineInput

        # Two prompts: first has images, second doesn't
        engine_input: InferenceEngineInput = {
            "prompts": None,
            "prompt_token_ids": [[1, 2], [3, 4]],
            "sampling_params": None,
            "session_ids": None,
            "multi_modal_data": [{"image": ["img1", "img2"]}, None],
        }

        assert len(engine_input["multi_modal_data"]) == 2
        assert engine_input["multi_modal_data"][0] is not None
        assert engine_input["multi_modal_data"][1] is None


# --- Integration test for image accumulation pattern ---


@pytest.mark.skipif(not PIL_AVAILABLE, reason="PIL not installed")
class TestImageAccumulationPattern:
    """Tests for the image accumulation pattern used in agent_loop."""

    def test_accumulate_images_across_steps(self, simple_red_image_base64):
        """Simulate accumulating images across multiple agent steps."""
        # Step 1: Initial prompt with one image
        step1_conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Look at this screenshot"},
                    {"type": "image_url", "image_url": {"url": simple_red_image_base64}},
                ],
            },
        ]

        accumulated_images = extract_images_from_conversation(step1_conversation)
        assert len(accumulated_images) == 1

        # Step 2: Environment returns another observation with image
        step2_observation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Here's what happened"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/step2.png"}},
                ],
            },
        ]

        new_images = extract_images_from_conversation(step2_observation)
        accumulated_images = accumulated_images + new_images  # Accumulate

        assert len(accumulated_images) == 2
        assert isinstance(accumulated_images[0], Image.Image)  # First was base64
        assert accumulated_images[1] == "https://example.com/step2.png"  # Second was URL

        # Step 3: Text-only observation (no new images)
        step3_observation = [
            {"role": "user", "content": "Task completed successfully"},
        ]

        new_images = extract_images_from_conversation(step3_observation)
        accumulated_images = accumulated_images + new_images

        # Should still have 2 images
        assert len(accumulated_images) == 2

    def test_multimodal_data_snapshot_per_step(self, simple_red_image_base64):
        """Each step should capture its own snapshot of accumulated images."""
        accumulated = []

        # Simulate adding images at different steps
        step1_images = extract_images_from_conversation(
            [
                {
                    "role": "user",
                    "content": [{"type": "image_url", "image_url": {"url": simple_red_image_base64}}],
                }
            ]
        )
        accumulated.extend(step1_images)
        step1_snapshot = list(accumulated)  # Copy for step 1

        step2_images = extract_images_from_conversation(
            [
                {
                    "role": "user",
                    "content": [{"type": "image_url", "image_url": {"url": "https://example.com/img2.png"}}],
                }
            ]
        )
        accumulated.extend(step2_images)
        step2_snapshot = list(accumulated)  # Copy for step 2

        # Verify snapshots are independent
        assert len(step1_snapshot) == 1
        assert len(step2_snapshot) == 2
        assert len(accumulated) == 2

        # Modifying step1_snapshot shouldn't affect step2_snapshot
        step1_snapshot.append("extra")
        assert len(step2_snapshot) == 2
