import cv2
import numpy as np
import pytesseract
from pytesseract import Output
import pyautogui
import torch
from transformers import MarianMTModel, MarianTokenizer
import logging

# Set up logging to file (or you could change it to console if you prefer)
logging.basicConfig(
    filename='ocr_korean_text_log.txt',  # Log file name
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# If needed, specify the Tesseract command path (adjust the path for your system)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Translation model (Korean to English)
model_name = "Helsinki-NLP/opus-mt-ko-en"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

def translate_ko_to_en(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
    translated = model.generate(**inputs)
    return tokenizer.decode(translated[0], skip_special_tokens=True)

def tesseract_ocr(image):
    # Convert image to grayscale for better OCR performance.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Define Tesseract configuration:
    # --oem 1 to force the LSTM OCR engine and --psm 6 for a uniform block of text.
    custom_config = r'--oem 1 --psm 6'
    
    # Get OCR data with bounding boxes.
    data = pytesseract.image_to_data(gray, lang='kor', config=custom_config, output_type=Output.DICT)
    
    # Group words by block number to combine individual words into text blocks.
    blocks = {}
    n_boxes = len(data['level'])
    for i in range(n_boxes):
        text = data['text'][i].strip()
        try:
            conf = int(data['conf'][i])
        except ValueError:
            continue
        if text and conf > 50:  # You may adjust the confidence threshold.
            block_id = data['block_num'][i]
            left = data['left'][i]
            top = data['top'][i]
            width = data['width'][i]
            height = data['height'][i]
            
            if block_id not in blocks:
                blocks[block_id] = {
                    'text': text,
                    'left': left,
                    'top': top,
                    'right': left + width,
                    'bottom': top + height
                }
            else:
                blocks[block_id]['text'] += " " + text
                blocks[block_id]['left'] = min(blocks[block_id]['left'], left)
                blocks[block_id]['top'] = min(blocks[block_id]['top'], top)
                blocks[block_id]['right'] = max(blocks[block_id]['right'], left + width)
                blocks[block_id]['bottom'] = max(blocks[block_id]['bottom'], top + height)
    
    # Convert the blocks dictionary to a list of (bounding_box, text) tuples.
    results = []
    for block in blocks.values():
        bbox = [(block['left'], block['top']), (block['right'], block['bottom'])]
        results.append((bbox, block['text']))
    return results

frame_count = 0
last_results = []

while True:
    # Capture a screenshot.
    screen = pyautogui.screenshot()
    screen_np = np.array(screen)
    h, w, _ = screen_np.shape
    # Crop to the middle 80% of the right half of the screen.
    cropped = screen_np[int(h * 0.1):int(h * 0.9), w // 2:]
    
    if cropped.size == 0:
        print("Cropped image is empty. Check cropping indices.")
        continue

    # Convert from RGB (PIL) to BGR (OpenCV).
    frame = cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR)

    frame_count += 1
    if frame_count % 60 == 0:
        ocr_results = tesseract_ocr(frame)
        last_results = []
        for (bbox, text) in ocr_results:
            # Log the detected Korean text to our log file.
            logging.info("Detected Korean text: %s", text)
            translated = translate_ko_to_en(text)
            last_results.append((bbox, translated))

    # Draw bounding boxes and translated text on the frame.
    for (bbox, translated) in last_results:
        (tl, br) = bbox
        text_position = (tl[0], max(0, tl[1] - 10))
        cv2.rectangle(frame, tl, br, (0, 255, 0), 2)
        cv2.putText(frame, translated, text_position,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("Translated", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
