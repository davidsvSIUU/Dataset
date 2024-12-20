import fitz
import base64
import tempfile
from typing import List, Tuple

def pdf_to_images(pdf_path: str) -> List[Tuple[int, str]]:
    try:
        pdf_document = fitz.open(pdf_path)
        images = []
        first_page = pdf_document[0]
        pix = first_page.get_pixmap()
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            pix.save(tmp_file.name)
            with open(tmp_file.name, 'rb') as image_file:
                first_page_b64 = base64.b64encode(image_file.read()).decode('utf-8')
                images.append((0, first_page_b64))
        total_pages = len(pdf_document)
        random_pages = range(1, total_pages)
        for page_num in random_pages:
            page = pdf_document[page_num]
            pix = page.get_pixmap()
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                pix.save(tmp_file.name)
                with open(tmp_file.name, 'rb') as image_file:
                    page_b64 = base64.b64encode(image_file.read()).decode('utf-8')
                    images.append((page_num, page_b64))
        return images
    except Exception as e:
        print(f"Error processing PDF {pdf_path}: {str(e)}")
        raise

def capture_page_image(pdf_path: str, page_number: int) -> bytes:
    try:
        pdf_document = fitz.open(pdf_path)
        page = pdf_document[page_number]
        pix = page.get_pixmap()
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            pix.save(tmp_file.name)
            with open(tmp_file.name, 'rb') as image_file:
                return image_file.read()
    except Exception as e:
        print(f"Error capturing page image: {str(e)}")
        return None

def capture_page_image_hd(pdf_path: str, page_number: int) -> bytes:
    try:
        pdf_document = fitz.open(pdf_path)
        page = pdf_document[page_number]
        
        # Increase the resolution/DPI
        zoom = 2.0  # Increase this value for higher resolution
        mat = fitz.Matrix(zoom, zoom)  # Create transformation matrix
        
        # Get the pixmap with enhanced settings
        pix = page.get_pixmap(
            matrix=mat,            # Apply zoom transformation
            alpha=False,           # Remove alpha channel for cleaner output
            colorspace=fitz.csRGB  # Ensure RGB colorspace
        )
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            pix.save(tmp_file.name)
            with open(tmp_file.name, 'rb') as image_file:
                return image_file.read()
    finally:
        if 'pdf_document' in locals():
            pdf_document.close()
        if 'tmp_file' in locals():
            try:
                os.unlink(tmp_file.name)
            except:
                pass

def capture_page_image_jpeg(pdf_path: str, page_number: int) -> bytes:
    try:
        pdf_document = fitz.open(pdf_path)
        page = pdf_document[page_number]
        pix = page.get_pixmap()
        with tempfile.NamedTemporaryFile(suffix='.jpeg', delete=False) as tmp_file:
            pix.save(tmp_file.name)
            with open(tmp_file.name, 'rb') as image_file:
                return image_file.read()
    except Exception as e:
        print(f"Error capturing page image: {str(e)}")
        return None