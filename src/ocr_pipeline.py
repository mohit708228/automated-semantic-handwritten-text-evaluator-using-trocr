import cv2
import numpy as np
from PIL import Image
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import re

# ---------------------------------------------------------
# 1. IMAGE PREPROCESSING
# ---------------------------------------------------------

def load_image(image_path):
    """Loads an image from the specified path using OpenCV."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not find or load image at {image_path}")
    return img


def _get_skew_angle(binary_img):
    """Estimates the skew angle from a binary (white-on-black) image."""
    coords = np.column_stack(np.where(binary_img > 0))
    if len(coords) < 50:
        return 0.0
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    return angle


def _apply_deskew(image, angle):
    """Rotates image by the given angle for deskewing."""
    if abs(angle) < 0.3 or abs(angle) > 15:
        return image
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


def create_binary_for_segmentation(gray):
    """
    Creates a binary image ONLY for line segmentation purposes.
    This is the 'map stream' — it finds where lines are.
    """
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Otsu automatically picks the best global threshold
    binary = cv2.threshold(blurred, 0, 255,
                           cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    # Light morphological close to connect broken strokes
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    return binary


def prepare_clean_gray(image):
    """
    Prepares a lightly processed grayscale image for TrOCR inference.
    This is the 'AI stream' — TrOCR works best with natural-looking images,
    so we only do minimal cleanup (no binarization, no heavy CLAHE).
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Very gentle CLAHE — just enough to normalize uneven lighting
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(16, 16))
    gray = clahe.apply(gray)

    return gray


def cv2_to_pil(cv2_img):
    """Converts an OpenCV Image to PIL format."""
    if len(cv2_img.shape) == 2:
        return Image.fromarray(cv2_img)
    return Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))


# ---------------------------------------------------------
# 2. LINE SEGMENTATION
# ---------------------------------------------------------

def segment_lines(binary_img, original_img):
    """
    Segments the image into individual text lines using horizontal projection profile.
    binary_img: binary image (text=white) used to FIND line coordinates.
    original_img: clean grayscale image to CROP lines from.
    """
    h, w = binary_img.shape[:2]

    # Horizontal projection profile
    hist = np.sum(binary_img, axis=1)

    # Smooth to avoid small gaps splitting lines
    kernel_size = max(5, h // 200)
    hist_smoothed = np.convolve(hist, np.ones(kernel_size) / kernel_size, mode='same')

    # Threshold: 5% of max — catches faint/light lines
    threshold = np.max(hist_smoothed) * 0.05

    in_line = False
    lines = []
    start_y = 0

    for y, val in enumerate(hist_smoothed):
        if val > threshold and not in_line:
            in_line = True
            start_y = y
        elif val <= threshold and in_line:
            in_line = False
            lines.append((start_y, y))

    # Handle text running to bottom edge
    if in_line:
        lines.append((start_y, h))

    # Merge ONLY pixel-noise splits (≤2px gap). Real handwriting gaps are 3px+.
    min_gap = 2

    merged_lines = []
    for line in lines:
        if merged_lines and line[0] - merged_lines[-1][1] <= min_gap:
            merged_lines[-1] = (merged_lines[-1][0], line[1])
        else:
            merged_lines.append(list(line))

    # Dynamic minimum line height
    min_line_height = max(15, h // 100)

    # Crop from the CLEAN original with generous padding
    padding = 12
    line_images = []
    for (start_y, end_y) in merged_lines:
        if end_y - start_y > min_line_height:
            y1 = max(0, start_y - padding)
            y2 = min(h, end_y + padding)
            line_img = original_img[y1:y2, :]
            line_images.append(line_img)

    return line_images


# ---------------------------------------------------------
# 3. TrOCR INFERENCE
# ---------------------------------------------------------

class TrOCRInferencer:
    """
    Wrapper for fine-tuned TrOCR model inference.
    """
    def __init__(self, model_path="my_trocr_model"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading Fine-Tuned TrOCR model from '{model_path}' on {self.device}...")

        self.processor = TrOCRProcessor.from_pretrained(model_path)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_path).to(self.device)
        self.model.eval()

        print("Fine-tuned model loaded successfully.")

    def predict_line(self, image_pil):
        """
        Runs inference on a single line image (PIL format).
        Returns (text, confidence_score).
        """
        if image_pil.mode != "RGB":
            image_pil = image_pil.convert("RGB")

        # --- Image preparation for TrOCR ---
        w, h = image_pil.size

        # 1. Scale to a reasonable height (128px matches IAM-style training)
        target_h = 128
        scale = target_h / h
        target_w = max(32, int(w * scale))  # ensure minimum width
        image_pil = image_pil.resize((target_w, target_h), Image.LANCZOS)

        # 2. Add horizontal white padding to prevent edge hallucinations
        pad = 24
        padded = Image.new("RGB", (target_w + pad * 2, target_h), (255, 255, 255))
        padded.paste(image_pil, (pad, 0))

        # 3. Ensure dark-text-on-light-background
        #    (TrOCR expects this; invert if the crop is mostly dark)
        arr = np.array(padded.convert("L"))
        if np.mean(arr) < 127:
            padded = Image.fromarray(255 - np.array(padded))

        # --- Inference ---
        pixel_values = self.processor(padded, return_tensors="pt").pixel_values.to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                pixel_values,
                num_beams=4,
                length_penalty=1.0,
                early_stopping=True,
                max_new_tokens=128,
                return_dict_in_generate=True,
                output_scores=True
            )

        generated_ids = outputs.sequences
        text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # --- Confidence score ---
        score = 0.0
        if hasattr(outputs, 'sequences_scores') and outputs.sequences_scores is not None:
            # sequences_scores is ALREADY length-normalized by HuggingFace beam search:
            #   score = sum(log_probs) / (length ^ length_penalty)
            # With length_penalty=1.0, this is the average log-prob per token.
            # Exponentiating gives the geometric mean of per-token probabilities.
            raw_score = torch.exp(outputs.sequences_scores[0]).item()
            score = min(max(raw_score, 0.0), 1.0)  # clamp to [0, 1]
        else:
            score = 1.0

        return text.strip(), score


# ---------------------------------------------------------
# 4. POST-PROCESSING
# ---------------------------------------------------------

def clean_text(text):
    """
    Lightweight cleanup of common TrOCR output artifacts:
    - Collapses multiple spaces.
    - Removes stray leading/trailing punctuation on each line.
    """
    # Collapse multiple spaces/tabs
    text = re.sub(r'[ \t]+', ' ', text)
    # Remove lines that are just punctuation or single chars (noise)
    lines = text.split('\n')
    cleaned = []
    for line in lines:
        stripped = line.strip()
        if len(stripped) > 1 or (len(stripped) == 1 and stripped.isalnum()):
            cleaned.append(stripped)
    return '\n'.join(cleaned).strip()


# ---------------------------------------------------------
# 5. FULL PIPELINE WRAPPER
# ---------------------------------------------------------

def extract_handwritten_text(image_path, model_path="my_trocr_model"):
    """
    End-to-end extraction using a Dual-Stream pipeline:
    - Binary stream: finds line coordinates (segmentation).
    - Grayscale stream: feeds clean crops to TrOCR (inference).
    """
    try:
        # 1. Load raw image
        img_cv2 = load_image(image_path)

        # 2. Prepare both streams from the raw image
        clean_gray = prepare_clean_gray(img_cv2)
        binary = create_binary_for_segmentation(clean_gray)

        # 3. Deskew both streams using the SAME angle
        skew_angle = _get_skew_angle(binary)
        binary = _apply_deskew(binary, skew_angle)
        clean_gray = _apply_deskew(clean_gray, skew_angle)

        # 4. Segment: find lines on binary, crop from clean gray
        lines_cv2 = segment_lines(binary, clean_gray)

        if not lines_cv2:
            print("[OCR] No lines detected in segmentation.")
            return "", 0.0

        # 5. Run TrOCR on each line crop
        inferencer = TrOCRInferencer(model_path=model_path)

        extracted_text_lines = []
        confidences = []

        for i, line_cv2 in enumerate(lines_cv2):
            line_pil = cv2_to_pil(line_cv2)
            text, conf = inferencer.predict_line(line_pil)

            if text:
                extracted_text_lines.append(text)
                confidences.append(conf)
                print(f"  Line {i+1}: conf={conf:.3f} | {text}")

        # 6. Assemble final output
        final_text = "\n".join(extracted_text_lines)
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
        return clean_text(final_text), avg_conf

    except Exception as e:
        print(f"Error during extraction: {e}")
        return "", 0.0

# Alias to maintain backwards compatibility with existing project
run_ocr_pipeline = extract_handwritten_text

if __name__ == "__main__":
    # Example usage:
    # result = extract_handwritten_text("img/sample.jpg")
    # print(result)
    pass