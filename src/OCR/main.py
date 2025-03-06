import os
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image


def extract_text_from_image(image_path):
    """
    Extract raw text from an image using GOT-OCR model.

    Args:
        image_path: Path to the image file

    Returns:
        Raw extracted text
    """

    # Set device
    device = "cpu"

    # Load model and processor
    print("Loading model and processor...")
    processor = AutoProcessor.from_pretrained("stepfun-ai/GOT-OCR-2.0-hf")
    model = AutoModelForImageTextToText.from_pretrained(
        "stepfun-ai/GOT-OCR-2.0-hf")

    # Read image
    print(f"Processing image: {image_path}")
    image = Image.open(image_path)

    # Process image
    inputs = processor(images=image, return_tensors="pt")

    # Generate predictions
    print("Generating text...")
    generate_ids = model.generate(
        **inputs,
        do_sample=False,
        tokenizer=processor.tokenizer,
        stop_strings="<|im_end|>",
        max_new_tokens=1024,
    )

    # Decode generated text
    raw_text = processor.decode(
        generate_ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)


# Main function
if __name__ == "__main__":
    # Choose an image to process
    image_path = "ktp/ktp1.png"  # Change this to your specific image

    print(f"\nProcessing image: {image_path}")
    try:
        raw_text = extract_text_from_image(image_path)
        print("\nExtracted Raw Text:")
        print(raw_text)

    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
