import torch
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\harik\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
from PIL import Image
from transformers import BlipProcessor , BlipForConditionalGeneration


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
).to(device)

def process_image_content(image: Image.Image) -> dict:
    inputs = processor(image, return_tensors="pt").to(device)
    output = model.generate(**inputs)
    caption = processor.decode(output[0], skip_special_tokens=True)

    ocr_text = pytesseract.image_to_string(image)

    return {
        "caption": caption.strip(),
        "ocr": ocr_text.strip()
    }