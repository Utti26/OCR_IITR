from byaldi import RAGMultiModalModel
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import gradio as gr 
from PIL import Image
import re


# Load models
def initialize_models():
    """Loads and returns the RAG multimodal and Qwen2-VL models along with the processor."""
    multimodal_rag = RAGMultiModalModel.from_pretrained("vidore/colpali")
    qwen_model = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", trust_remote_code=True, torch_dtype=torch.float32)
    qwen_processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", trust_remote_code=True)
    return multimodal_rag, qwen_model, qwen_processor

multimodal_rag, qwen_model, qwen_processor = initialize_models()

# Text extraction function
def perform_ocr(image):
    """Extracts Sanskrit and English text from an image using the Qwen model."""
    query = "Extract text from the image in original language"

    # Format the request for the model
    user_input = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": query}
            ]
        }
    ]

    # Preprocess the input
    input_text = qwen_processor.apply_chat_template(user_input, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(user_input)
    model_inputs = qwen_processor(
        text=[input_text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
    ).to("cpu")  # Use CPU for inference

    # Generate output
    with torch.no_grad():
        generated_ids = qwen_model.generate(**model_inputs, max_new_tokens=2000)
        trimmed_ids = [output[len(input_ids):] for input_ids, output in zip(model_inputs.input_ids, generated_ids)]
        ocr_result = qwen_processor.batch_decode(trimmed_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    return ocr_result

# Keyword search function
def highlight_keyword(text, keyword):
    """Searches and highlights the keyword in the extracted text."""
    keyword_lowercase = keyword.lower()
    sentences = text.split('. ')
    results = []

    for sentence in sentences:
        if keyword_lowercase in sentence.lower():
            highlighted = re.sub(f'({re.escape(keyword)})', r'<mark>\1</mark>', sentence, flags=re.IGNORECASE)
            results.append(highlighted)

    return results if results else ["No matches found."]

# Gradio app for text extraction
def extract_text(image):
    """Extracts text from an uploaded image."""
    return perform_ocr(image)

# Gradio app for keyword search
def search_in_text(extracted_text, keyword):
    """Searches for a keyword in the extracted text and highlights matches."""
    results = highlight_keyword(extracted_text, keyword)
    return "<br>".join(results)

# Updated title with revised phrasing
header_html = """
<h1 style="text-align: center; color: #4CAF50;"><span class="gradient-text">OCR and Text Search Prototype</span></h1>
"""

# CSS to fix button sizes
custom_css = """
    .gr-button {
        width: 200px; /* Set a fixed width for the buttons */
        padding: 12px 20px; /* Add padding to buttons for consistency */
    }
    .gr-textbox {
        max-height: 300px; /* Set a maximum height for the extracted text output */
        overflow-y: scroll; /* Enable scrolling when text exceeds the height */
    }
"""

# Gradio Interface
with gr.Blocks(css=custom_css) as interface:

    # Header section
    gr.HTML(header_html)

    # Sidebar section
    with gr.Row():
        with gr.Column(scale=1, min_width=200):
            gr.Markdown("## Instructions")
            gr.Markdown("""
                1. Upload an image containing text.
                2. Extract the text from the image.
                3. Search for specific keywords within the extracted text.
            """)
            gr.Markdown("### Features")
            gr.Markdown("""
                - **OCR**: Extract text from images.
                - **Keyword Search**: Search and highlight keywords in extracted text.
            """)

        with gr.Column(scale=3):
            # Main content in tabs
            with gr.Tabs():

                # First Tab: Text Extraction
                with gr.Tab("Extract Text"):
                    gr.Markdown("### Upload an image to extract text:")
                    with gr.Row():
                        image_upload = gr.Image(type="pil", label="Upload Image", interactive=True)
                    with gr.Row():
                        extract_btn = gr.Button("Extract Text")
                        extracted_textbox = gr.Textbox(label="Extracted Text")
                    extract_btn.click(extract_text, inputs=image_upload, outputs=extracted_textbox)

                # Second Tab: Keyword Search
                with gr.Tab("Search in Extracted Text"):
                    gr.Markdown("### Search for a keyword in the extracted text:")
                    with gr.Row():
                        keyword_searchbox = gr.Textbox(label="Enter Keyword", placeholder="Keyword to search")
                    with gr.Row():
                        search_btn = gr.Button("Search")
                        search_results = gr.HTML(label="Results")
                    search_btn.click(search_in_text, inputs=[extracted_textbox, keyword_searchbox], outputs=search_results)

# Launch the Gradio App
interface.launch()
