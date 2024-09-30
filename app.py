import gradio as gr
import spaces
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, TextIteratorStreamer
from qwen_vl_utils import process_vision_info
import torch
from PIL import Image
import subprocess
import numpy as np
import os
from threading import Thread
import uuid
import io
import re  # Import regular expressions for word highlighting

# Model and Processor Loading (Done once at startup)
MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    torch_dtype=torch.float16
).to("cuda").eval()
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

DESCRIPTION = "[Qwen2-VL-2B Demo](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct)"

# Define supported media extensions
image_extensions = Image.registered_extensions()
video_extensions = ("avi", "mp4", "mov", "mkv", "flv", "wmv", "mjpeg", "wav", "gif", "webm", "m4v", "3gp")


def identify_and_save_blob(blob_path):
    """Identifies if the blob is an image or video and saves it accordingly."""
    try:
        with open(blob_path, 'rb') as file:
            blob_content = file.read()

            # Try to identify if it's an image
            try:
                Image.open(io.BytesIO(blob_content)).verify()  # Check if it's a valid image
                extension = ".png"  # Default to PNG for saving
                media_type = "image"
            except (IOError, SyntaxError):
                # If it's not a valid image, assume it's a video
                extension = ".mp4"  # Default to MP4 for saving
                media_type = "video"

            # Create a unique filename
            filename = f"temp_{uuid.uuid4()}_media{extension}"
            with open(filename, "wb") as f:
                f.write(blob_content)

            return filename, media_type

    except FileNotFoundError:
        raise ValueError(f"The file {blob_path} was not found.")
    except Exception as e:
        raise ValueError(f"An error occurred while processing the file: {e}")


@spaces.GPU
def qwen_inference(media_input, search_word):
    """
    Performs OCR on the input media and highlights the search_word in the extracted text.

    Args:
        media_input (str): Path to the uploaded image or video file.
        search_word (str): The word to search and highlight in the OCR result.

    Yields:
        str: The OCR result with highlighted search words.
    """
    text_input = "Extract text"  # Hardcoded text query

    if isinstance(media_input, str):  # If it's a filepath
        media_path = media_input
        if media_path.endswith(tuple([i for i, f in image_extensions.items()])):
            media_type = "image"
        elif media_path.endswith(video_extensions):
            media_type = "video"
        else:
            try:
                media_path, media_type = identify_and_save_blob(media_input)
                print(media_path, media_type)
            except Exception as e:
                print(e)
                raise ValueError(
                    "Unsupported media type. Please upload an image or video."
                )

    print(f"Processing media: {media_path} (Type: {media_type})")

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": media_type,
                    media_type: media_path,
                    **({"fps": 8.0} if media_type == "video" else {}),
                },
                {"type": "text", "text": text_input},
            ],
        }
    ]

    # Apply chat template to format the input for the model
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)

    # Prepare model inputs
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    # Initialize the streamer for iterative generation
    streamer = TextIteratorStreamer(
        processor, skip_prompt=True, **{"skip_special_tokens": True}
    )
    generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=1024)

    # Start the generation in a separate thread
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    buffer = ""
    for new_text in streamer:
        buffer += new_text
        # Highlight the search_word in the buffer
        if search_word:
            # Use regex for case-insensitive search and highlight
            pattern = re.compile(re.escape(search_word), re.IGNORECASE)
            highlighted_text = pattern.sub(lambda m: f"<mark>{m.group(0)}</mark>", buffer)
        else:
            highlighted_text = buffer
        yield highlighted_text


css = """
  #output {
    height: 500px;
    overflow: auto;
    border: 1px solid #ccc;
  }
"""

with gr.Blocks(css=css) as demo:
    gr.Markdown(DESCRIPTION)

    with gr.Tab(label="Image/Video Input"):
        with gr.Row():
            with gr.Column():
                input_media = gr.File(
                    label="Upload Image or Video", type="filepath"
                )
                search_word = gr.Textbox(
                    label="Search Word", placeholder="Enter word to highlight", lines=1
                )
                submit_btn = gr.Button(value="Submit")
            with gr.Column():
                # Use HTML component to display highlighted text
                output_text = gr.HTML(label="Output Text")

        submit_btn.click(
            qwen_inference,
            inputs=[input_media, search_word],
            outputs=[output_text]
        )

demo.launch(debug=True)
