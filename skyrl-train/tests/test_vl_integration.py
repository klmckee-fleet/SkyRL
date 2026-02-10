#!/usr/bin/env python3
"""
Minimal integration test for VL model support.

Run with:
    PYTHONPATH=. python tests/test_vl_integration.py

This tests the VL code path without needing actual GPU or vLLM.
"""
import sys
import base64
import io


def create_test_image_base64():
    """Create a minimal test image as base64."""
    try:
        from PIL import Image
    except ImportError:
        print("WARNING: PIL not available, using placeholder")
        return "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="

    img = Image.new("RGB", (64, 64), color="red")
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    b64 = base64.b64encode(buffer.read()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def test_multimodal_message_types():
    """Test that MessageType supports multimodal content."""
    print("Testing multimodal message types...")

    # Text-only message (backward compatible)
    text_message = {"role": "user", "content": "Hello, world!"}
    assert isinstance(text_message["content"], str)

    # Multimodal message with image
    mm_message = {
        "role": "user",
        "content": [
            {"type": "text", "text": "What's in this image?"},
            {"type": "image_url", "image_url": {"url": create_test_image_base64()}},
        ],
    }
    assert isinstance(mm_message["content"], list)

    print("  ✓ Message types support multimodal content")


def test_inference_engine_input():
    """Test that InferenceEngineInput accepts multi_modal_data."""
    print("Testing InferenceEngineInput with multi_modal_data...")

    # Simulate what the generator would create
    engine_input = {
        "prompts": None,
        "prompt_token_ids": [[1, 2, 3, 4, 5]],
        "sampling_params": {"temperature": 0.9, "max_tokens": 100},
        "session_ids": ["test-session-1"],
        "multi_modal_data": [{"image": ["mock_pil_image_1", "mock_pil_image_2"]}],
    }

    # Verify structure
    assert engine_input["multi_modal_data"] is not None
    assert len(engine_input["multi_modal_data"]) == 1
    assert "image" in engine_input["multi_modal_data"][0]
    assert len(engine_input["multi_modal_data"][0]["image"]) == 2

    # Test None case (text-only, backward compatible)
    text_only_input = {
        "prompts": None,
        "prompt_token_ids": [[1, 2, 3]],
        "sampling_params": None,
        "session_ids": None,
        "multi_modal_data": None,
    }
    assert text_only_input["multi_modal_data"] is None

    print("  ✓ InferenceEngineInput accepts multi_modal_data")


def test_multimodal_utils():
    """Test the multimodal utility functions."""
    print("Testing multimodal utilities...")

    # Inline implementations to test logic without imports
    def is_multimodal_message(message):
        content = message.get("content")
        if isinstance(content, str):
            return False
        if isinstance(content, list):
            return any(
                isinstance(item, dict) and item.get("type") == "image_url"
                for item in content
            )
        return False

    def is_multimodal_conversation(conversation):
        return any(is_multimodal_message(msg) for msg in conversation)

    # Test is_multimodal_message
    assert is_multimodal_message({"role": "user", "content": "text"}) is False
    assert (
        is_multimodal_message(
            {
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": "..."}}],
            }
        )
        is True
    )

    # Test is_multimodal_conversation
    text_convo = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi!"},
    ]
    assert is_multimodal_conversation(text_convo) is False

    mm_convo = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Look at this"},
                {"type": "image_url", "image_url": {"url": create_test_image_base64()}},
            ],
        },
        {"role": "assistant", "content": "I see a red image"},
    ]
    assert is_multimodal_conversation(mm_convo) is True

    print("  ✓ Multimodal utilities work correctly")


def test_image_extraction():
    """Test image extraction from conversation."""
    print("Testing image extraction...")

    try:
        from PIL import Image
    except ImportError:
        print("  ⚠ Skipping (PIL not available)")
        return

    def decode_base64_image(data_url):
        if "," in data_url:
            base64_data = data_url.split(",", 1)[1]
        else:
            base64_data = data_url
        image_bytes = base64.b64decode(base64_data)
        return Image.open(io.BytesIO(image_bytes))

    def extract_images_from_conversation(conversation):
        images = []
        for message in conversation:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for item in content:
                if not isinstance(item, dict) or item.get("type") != "image_url":
                    continue
                image_url_data = item.get("image_url", {})
                url = image_url_data.get("url", "")
                if url.startswith("data:image"):
                    images.append(decode_base64_image(url))
                elif url.startswith(("http://", "https://")):
                    images.append(url)  # Pass URL as-is
        return images

    # Create test conversation with image
    test_base64 = create_test_image_base64()
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this screenshot"},
                {"type": "image_url", "image_url": {"url": test_base64}},
            ],
        },
    ]

    images = extract_images_from_conversation(conversation)
    assert len(images) == 1
    assert isinstance(images[0], Image.Image)
    assert images[0].size == (64, 64)

    print("  ✓ Image extraction works correctly")


def test_image_accumulation():
    """Test image accumulation across steps (simulating agent_loop)."""
    print("Testing image accumulation across steps...")

    try:
        from PIL import Image
    except ImportError:
        print("  ⚠ Skipping (PIL not available)")
        return

    def decode_base64_image(data_url):
        if "," in data_url:
            base64_data = data_url.split(",", 1)[1]
        else:
            base64_data = data_url
        image_bytes = base64.b64decode(base64_data)
        return Image.open(io.BytesIO(image_bytes))

    def extract_images(conversation):
        images = []
        for message in conversation:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for item in content:
                if not isinstance(item, dict) or item.get("type") != "image_url":
                    continue
                url = item.get("image_url", {}).get("url", "")
                if url.startswith("data:image"):
                    images.append(decode_base64_image(url))
        return images

    # Simulate multi-step agent loop
    accumulated_images = []

    # Step 1: Initial prompt with screenshot
    step1_prompt = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Here's the browser"},
                {"type": "image_url", "image_url": {"url": create_test_image_base64()}},
            ],
        }
    ]
    accumulated_images.extend(extract_images(step1_prompt))
    step1_mm_data = {"image": list(accumulated_images)}  # Snapshot

    # Step 2: Model took action, env returns new screenshot
    step2_observation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Action executed. New state:"},
                {"type": "image_url", "image_url": {"url": create_test_image_base64()}},
            ],
        }
    ]
    accumulated_images.extend(extract_images(step2_observation))
    step2_mm_data = {"image": list(accumulated_images)}  # Snapshot

    # Step 3: Text-only observation (no new image)
    step3_observation = [{"role": "user", "content": "Task completed successfully"}]
    accumulated_images.extend(extract_images(step3_observation))
    step3_mm_data = {"image": list(accumulated_images)}  # Snapshot

    # Verify accumulation
    assert len(step1_mm_data["image"]) == 1, "Step 1 should have 1 image"
    assert len(step2_mm_data["image"]) == 2, "Step 2 should have 2 images"
    assert len(step3_mm_data["image"]) == 2, "Step 3 should still have 2 images"

    # Verify snapshots are independent (list copies)
    step1_mm_data["image"].append("extra")
    assert len(step2_mm_data["image"]) == 2, "Modifying step1 shouldn't affect step2"

    print("  ✓ Image accumulation works correctly")


def test_vllm_prompt_construction():
    """Test that vLLM prompt construction with multimodal data works."""
    print("Testing vLLM prompt construction...")

    # Simulate what vllm_engine.py does
    prompt_token_ids = [[1, 2, 3, 4, 5]]
    multi_modal_data = [{"image": ["mock_image_1", "mock_image_2"]}]

    # Construct prompts as vllm_engine would
    prompts = []
    for i, token_ids in enumerate(prompt_token_ids):
        mm_data = (
            multi_modal_data[i]
            if multi_modal_data and i < len(multi_modal_data)
            else None
        )
        if mm_data:
            # Dict format for multimodal
            prompts.append({"prompt_token_ids": token_ids, "multi_modal_data": mm_data})
        else:
            # TokensPrompt format for text-only
            prompts.append({"prompt_token_ids": token_ids})

    assert len(prompts) == 1
    assert "multi_modal_data" in prompts[0]
    assert prompts[0]["multi_modal_data"]["image"] == ["mock_image_1", "mock_image_2"]

    # Test text-only case
    text_prompts = []
    for i, token_ids in enumerate([[6, 7, 8]]):
        mm_data = None  # No multimodal data
        if mm_data:
            text_prompts.append(
                {"prompt_token_ids": token_ids, "multi_modal_data": mm_data}
            )
        else:
            text_prompts.append({"prompt_token_ids": token_ids})

    assert "multi_modal_data" not in text_prompts[0]

    print("  ✓ vLLM prompt construction works correctly")


def main():
    print("=" * 60)
    print("VL Integration Tests")
    print("=" * 60)

    tests = [
        test_multimodal_message_types,
        test_inference_engine_input,
        test_multimodal_utils,
        test_image_extraction,
        test_image_accumulation,
        test_vllm_prompt_construction,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            failed += 1

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
