import fitz
from PIL import Image
import io
import pdfplumber
import pandas as pd


def embed_pages(file):
    chunks = []
    file.seek(0)
    fitz_doc = fitz.open(stream=file.read(), filetype="pdf")
    file.seek(0)
    plumber_pdf = pdfplumber.open(file)
    total_pages = len(fitz_doc)
    for i in range(total_pages):
        text = fitz_doc[i].get_text().strip()
        chunks.append({"type": "text", "content": text, "page": i})
        images = fitz_doc[i].get_images(full=True)
        for idx,img in enumerate(images):
            xref = img[0]
            base_image = fitz_doc.extract_image(xref)
            image_bytes = base_image["image"]
            img_pil = Image.open(io.BytesIO(image_bytes))
            chunks.append({
                "type": "image",
                "content": img_pil,
                "page": i,
                "position": idx
            })
        if i < len(plumber_pdf.pages):
            tables = plumber_pdf.pages[i].extract_tables()
            for table in tables:
                df = pd.DataFrame(table)
                if df.shape[1] >= 2 and df.shape[0] > 1:
                    chunks.append({
                        "type": "table",
                        "content": df.to_csv(index=False),
                        "page": i
                    })
    fitz_doc.close()
    plumber_pdf.close()

    return chunks