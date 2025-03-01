#pdf_utils.py
import os 
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
    pdf_document = None
    tmp_file = None
    try:
        pdf_document = fitz.open(pdf_path)
        page = pdf_document[page_number]
        
        # Increase the resolution/DPI
        zoom = 2.0
        mat = fitz.Matrix(zoom, zoom)
        
        # Get the pixmap with enhanced settings
        pix = page.get_pixmap(
            matrix=mat,
            alpha=False,
            colorspace=fitz.csRGB
        )
        
        tmp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        tmp_file_name = tmp_file.name
        tmp_file.close()  # Close the file handle explicitly
        
        pix.save(tmp_file_name)
        with open(tmp_file_name, 'rb') as image_file:
            image_data = image_file.read()
        
        # Essayer de supprimer le fichier, mais ignorer les erreurs
        try:
            os.unlink(tmp_file_name)
        except Exception as e:
            print(f"Warning: Could not delete temporary file {tmp_file_name}: {e}")
            pass
            
        return image_data
    
    finally:
        if pdf_document:
            pdf_document.close()
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