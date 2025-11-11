# barcode_scan_helpers.py
from pyzbar.pyzbar import decode
from PIL import Image
import numpy as np
import io

def decode_barcodes_from_bytes(img_bytes):
    """
    Accepts image bytes (from Streamlit st.camera_input or st.file_uploader),
    returns list of decoded barcode/QR strings (may be empty).
    """
    try:
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception:
        return []
    arr = np.array(image)
    decoded = decode(arr)
    results = []
    for d in decoded:
        try:
            data = d.data.decode("utf-8")
        except Exception:
            data = d.data
        results.append({"data": data, "type": d.type})
    return results
