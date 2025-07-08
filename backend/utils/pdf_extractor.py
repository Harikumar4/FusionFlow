import fitz
from PIL import Image
import io
import pdfplumber
import pandas as pd

def extract_text_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def extract_images_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    images = []
    for page_index in range(len(doc)):
        for img_index, img in enumerate(doc[page_index].get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            img_pil = Image.open(io.BytesIO(image_bytes))
            images.append(img_pil)
    return images

def extract_tables_pdf(file):
    tables = []
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            extracted = page.extract_tables()
            for table in extracted:
                df = pd.DataFrame(table)
                if df.shape[1] >= 2 and df.shape[0] > 1:
                    tables.append(df)
    return tables
