import gradio as gr
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, TextIteratorStreamer
from qwen_vl_utils import process_vision_info
import torch
from PIL import Image
import io
import re
import uuid
from threading import Thread

# Model loading
MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    torch_dtype=torch.float16
).to("cuda").eval()
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

# Supported media extensions
image_extensions = Image.registered_extensions()
video_extensions = ("avi", "mp4", "mov", "mkv", "flv", "wmv", "mjpeg", "wav", "gif", "webm", "m4v", "3gp")

# Function to handle media processing
def identify_and_save_blob(blob_path):
    try:
        with open(blob_path, 'rb') as file:
            blob_content = file.read()
            try:
                Image.open(io.BytesIO(blob_content)).verify()
                extension = ".png"
                media_type = "image"
            except (IOError, SyntaxError):
                extension = ".mp4"
                media_type = "video"
            filename = f"temp_{uuid.uuid4()}_media{extension}"
            with open(filename, "wb") as f:
                f.write(blob_content)
            return filename, media_type
    except FileNotFoundError:
        raise ValueError(f"The file {blob_path} was not found.")
    except Exception as e:
        raise ValueError(f"An error occurred while processing the file: {e}")

# OCR and word search
def qwen_inference(media_input, search_word):
    text_input = "Extract text"
    if isinstance(media_input, str):
        media_path = media_input
        if media_path.endswith(tuple([i for i, f in image_extensions.items()])):
            media_type = "image"
        elif media_path.endswith(video_extensions):
            media_type = "video"
        else:
            try:
                media_path, media_type = identify_and_save_blob(media_input)
            except Exception as e:
                raise ValueError("Unsupported media type. Please upload an image or video.")
    messages = [{"role": "user", "content": [{"type": media_type, media_type: media_path, **({"fps": 8.0} if media_type == "video" else {})}, {"type": "text", "text": text_input}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to("cuda")
    streamer = TextIteratorStreamer(processor, skip_prompt=True, **{"skip_special_tokens": True})
    generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=1024)
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    buffer = ""
    for new_text in streamer:
        buffer += new_text
        if search_word:
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
    gr.Markdown("[Qwen2-VL-2B Demo](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct)")
    with gr.Tab(label="Image/Video Input"):
        with gr.Row():
            with gr.Column():
                input_media = gr.File(label="Upload Image or Video", type="filepath")
                search_word = gr.Textbox(label="Search Word", placeholder="Enter word to highlight", lines=1)
                submit_btn = gr.Button(value="Submit")
            with gr.Column():
                output_text = gr.HTML(label="Output Text")
        submit_btn.click(qwen_inference, inputs=[input_media, search_word], outputs=[output_text])
demo.launch(debug=True)
