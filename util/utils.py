# from ultralytics import YOLO
import os
os.environ['FLAGS_enable_pir_api'] = '0'
os.environ['FLAGS_enable_onednn'] = '0'
os.environ['FLAGS_enable_new_executor'] = '0'
os.environ['FLAGS_enable_executor_for_pir'] = '0'
import io
import base64
import time
from PIL import Image, ImageDraw, ImageFont
import json
import requests
# utility function
import os
from openai import AzureOpenAI

import json
import sys
import os
import cv2
import numpy as np

os.environ['FLAGS_allocator_strategy'] = 'auto_growth'
os.environ['FLAGS_fraction_of_gpu_memory_to_use'] = '0.1'
os.environ['FLAGS_selected_gpus'] = '0'
os.environ['FLAGS_cudnn_deterministic'] = 'True'

# %matplotlib inline
from matplotlib import pyplot as plt
import easyocr
from paddleocr import PaddleOCR

# Initialisierung von EasyOCR (bleibt unver�ndert)
reader = easyocr.Reader(['en', 'de'])

# KORRIGIERTE Initialisierung von PaddleOCR gem�� GitHub-Doku
# Wir behalten nur die Parameter, die offiziell unterst�tzt werden
paddle_ocr = PaddleOCR(
    lang='en',                # Unterst�tzt laut Quickstart
    use_angle_cls=False,      # Unterst�tzt laut Inferenz-Guide
    rec_batch_num=64,
    #det_model_dir='weights/paddle/det',
    #rec_model_dir='weights/paddle/rec',
    #cls_model_dir='weights/paddle/cls',
)



import time
import base64

import os
import ast
import torch
from typing import Tuple, List, Union
from torchvision.ops import box_convert
import re
from torchvision.transforms import ToPILImage
import supervision as sv
import torchvision.transforms as T
from util.box_annotator import BoxAnnotator 

import sys
from unittest.mock import MagicMock
import importlib.util
def mock_flash_attn():
    # Wir erstellen ein echtes Modul-Objekt statt nur eines Mocks
    module_name = 'flash_attn'
    spec = importlib.util.spec_from_loader(module_name, loader=None)
    flash_attn = importlib.util.module_from_spec(spec)
    
    # Wir f�gen die notwendigen Attribute hinzu, damit find_spec nicht abst�rzt
    flash_attn.__spec__ = spec
    flash_attn.flash_attn_2_beam_search = MagicMock()
    
    # Wir registrieren das Modul und seine Unterpfade im System
    sys.modules[module_name] = flash_attn
    sys.modules["flash_attn.flash_attn_interface"] = MagicMock()
    sys.modules["flash_attn.ops"] = MagicMock()
    sys.modules["flash_attn.layers"] = MagicMock()
    sys.modules["flash_attn.layers.rotary"] = MagicMock()

# Falls flash_attn nicht installiert ist (was unter Windows der Fall ist)
if importlib.util.find_spec("flash_attn") is None:
    mock_flash_attn()


def get_caption_model_processor(model_name, model_name_or_path="Salesforce/blip2-opt-2.7b", device=None):
    if not device:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if model_name == "blip2":
        from transformers import Blip2Processor, Blip2ForConditionalGeneration
        processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        if device == 'cpu':
            model = Blip2ForConditionalGeneration.from_pretrained(
            model_name_or_path, device_map=None, torch_dtype=torch.float32
        ) 
        else:
            model = Blip2ForConditionalGeneration.from_pretrained(
            model_name_or_path, device_map=None, torch_dtype=torch.float16
        ).to(device)
    elif model_name == "florence2":
        from transformers import AutoProcessor, AutoModelForCausalLM 
        processor = AutoProcessor.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True)
        if device == 'cpu':
            model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.float32, trust_remote_code=True, attn_implementation="sdpa")
        else:
            model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.float16, trust_remote_code=True, attn_implementation="sdpa").to(device)
    return {'model': model.to(device), 'processor': processor}


def get_yolo_model(model_path):
    from ultralytics import YOLO
    # Load the model.
    model = YOLO(model_path)
    return model


@torch.inference_mode()
def get_parsed_content_icon(filtered_boxes, starting_idx, image_source, caption_model_processor, prompt=None, batch_size=128):
    # Number of samples per batch, --> 128 roughly takes 4 GB of GPU memory for florence v2 model
    to_pil = ToPILImage()
    if starting_idx:
        non_ocr_boxes = filtered_boxes[starting_idx:]
    else:
        non_ocr_boxes = filtered_boxes
    croped_pil_image = []
    for i, coord in enumerate(non_ocr_boxes):
        try:
            xmin, xmax = int(coord[0]*image_source.shape[1]), int(coord[2]*image_source.shape[1])
            ymin, ymax = int(coord[1]*image_source.shape[0]), int(coord[3]*image_source.shape[0])
            cropped_image = image_source[ymin:ymax, xmin:xmax, :]
            cropped_image = cv2.resize(cropped_image, (64, 64))
            croped_pil_image.append(to_pil(cropped_image))
        except:
            continue

    model, processor = caption_model_processor['model'], caption_model_processor['processor']
    if not prompt:
        if 'florence' in model.config.name_or_path:
            prompt = "<CAPTION>"
        else:
            prompt = "The image shows"
    
    generated_texts = []
    device = model.device
    for i in range(0, len(croped_pil_image), batch_size):
        start = time.time()
        batch = croped_pil_image[i:i+batch_size]
        t1 = time.time()
        if model.device.type == 'cuda':
            inputs = processor(images=batch, text=[prompt]*len(batch), return_tensors="pt", do_resize=False).to(device=device, dtype=torch.float16)
        else:
            inputs = processor(images=batch, text=[prompt]*len(batch), return_tensors="pt").to(device=device)
        if 'florence' in model.config.name_or_path:
            generated_ids = model.generate(input_ids=inputs["input_ids"],pixel_values=inputs["pixel_values"],max_new_tokens=20,num_beams=1, do_sample=False)
        else:
            generated_ids = model.generate(**inputs, max_length=100, num_beams=5, no_repeat_ngram_size=2, early_stopping=True, num_return_sequences=1) # temperature=0.01, do_sample=True,
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
        generated_text = [gen.strip() for gen in generated_text]
        generated_texts.extend(generated_text)
    
    return generated_texts



def get_parsed_content_icon_phi3v(filtered_boxes, ocr_bbox, image_source, caption_model_processor):
    to_pil = ToPILImage()
    if ocr_bbox:
        non_ocr_boxes = filtered_boxes[len(ocr_bbox):]
    else:
        non_ocr_boxes = filtered_boxes
    croped_pil_image = []
    for i, coord in enumerate(non_ocr_boxes):
        xmin, xmax = int(coord[0]*image_source.shape[1]), int(coord[2]*image_source.shape[1])
        ymin, ymax = int(coord[1]*image_source.shape[0]), int(coord[3]*image_source.shape[0])
        cropped_image = image_source[ymin:ymax, xmin:xmax, :]
        croped_pil_image.append(to_pil(cropped_image))

    model, processor = caption_model_processor['model'], caption_model_processor['processor']
    device = model.device
    messages = [{"role": "user", "content": "<|image_1|>\ndescribe the icon in one sentence"}] 
    prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    batch_size = 5  # Number of samples per batch
    generated_texts = []

    for i in range(0, len(croped_pil_image), batch_size):
        images = croped_pil_image[i:i+batch_size]
        image_inputs = [processor.image_processor(x, return_tensors="pt") for x in images]
        inputs ={'input_ids': [], 'attention_mask': [], 'pixel_values': [], 'image_sizes': []}
        texts = [prompt] * len(images)
        for i, txt in enumerate(texts):
            input = processor._convert_images_texts_to_inputs(image_inputs[i], txt, return_tensors="pt")
            inputs['input_ids'].append(input['input_ids'])
            inputs['attention_mask'].append(input['attention_mask'])
            inputs['pixel_values'].append(input['pixel_values'])
            inputs['image_sizes'].append(input['image_sizes'])
        max_len = max([x.shape[1] for x in inputs['input_ids']])
        for i, v in enumerate(inputs['input_ids']):
            inputs['input_ids'][i] = torch.cat([processor.tokenizer.pad_token_id * torch.ones(1, max_len - v.shape[1], dtype=torch.long), v], dim=1)
            inputs['attention_mask'][i] = torch.cat([torch.zeros(1, max_len - v.shape[1], dtype=torch.long), inputs['attention_mask'][i]], dim=1)
        inputs_cat = {k: torch.concatenate(v).to(device) for k, v in inputs.items()}

        generation_args = { 
            "max_new_tokens": 25, 
            "temperature": 0.01, 
            "do_sample": False, 
        } 
        generate_ids = model.generate(**inputs_cat, eos_token_id=processor.tokenizer.eos_token_id, **generation_args) 
        # # remove input tokens 
        generate_ids = generate_ids[:, inputs_cat['input_ids'].shape[1]:]
        response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        response = [res.strip('\n').strip() for res in response]
        generated_texts.extend(response)

    return generated_texts

def remove_overlap(boxes, iou_threshold, ocr_bbox=None):
    if ocr_bbox is None:
        ocr_bbox = []
        
    # Die Assertion sollte jetzt nur noch sicherstellen, dass es eine Liste ist
    assert isinstance(ocr_bbox, list), "ocr_bbox muss eine Liste sein (ggf. leer)"

    def box_area(box):
        return (box[2] - box[0]) * (box[3] - box[1])

    def intersection_area(box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        return max(0, x2 - x1) * max(0, y2 - y1)

    def IoU(box1, box2):
        intersection = intersection_area(box1, box2)
        union = box_area(box1) + box_area(box2) - intersection + 1e-6
        if box_area(box1) > 0 and box_area(box2) > 0:
            ratio1 = intersection / box_area(box1)
            ratio2 = intersection / box_area(box2)
        else:
            ratio1, ratio2 = 0, 0
        return max(intersection / union, ratio1, ratio2)

    def is_inside(box1, box2):
        # return box1[0] >= box2[0] and box1[1] >= box2[1] and box1[2] <= box2[2] and box1[3] <= box2[3]
        intersection = intersection_area(box1, box2)
        ratio1 = intersection / box_area(box1)
        return ratio1 > 0.95

    boxes = boxes.tolist()
    filtered_boxes = []
    if ocr_bbox:
        filtered_boxes.extend(ocr_bbox)
    # print('ocr_bbox!!!', ocr_bbox)
    for i, box1 in enumerate(boxes):
        # if not any(IoU(box1, box2) > iou_threshold and box_area(box1) > box_area(box2) for j, box2 in enumerate(boxes) if i != j):
        is_valid_box = True
        for j, box2 in enumerate(boxes):
            # keep the smaller box
            if i != j and IoU(box1, box2) > iou_threshold and box_area(box1) > box_area(box2):
                is_valid_box = False
                break
        if is_valid_box:
            # add the following 2 lines to include ocr bbox
            if ocr_bbox:
                # only add the box if it does not overlap with any ocr bbox
                if not any(IoU(box1, box3) > iou_threshold and not is_inside(box1, box3) for k, box3 in enumerate(ocr_bbox)):
                    filtered_boxes.append(box1)
            else:
                filtered_boxes.append(box1)
    return torch.tensor(filtered_boxes)


def remove_overlap_new(boxes, iou_threshold, ocr_bbox=None):
    '''
    ocr_bbox format: [{'type': 'text', 'bbox':[x,y], 'interactivity':False, 'content':str }, ...]
    boxes format: [{'type': 'icon', 'bbox':[x,y], 'interactivity':True, 'content':None }, ...]

    '''
    assert ocr_bbox is None or isinstance(ocr_bbox, List)

    def box_area(box):
        return (box[2] - box[0]) * (box[3] - box[1])

    def intersection_area(box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        return max(0, x2 - x1) * max(0, y2 - y1)

    def IoU(box1, box2):
        intersection = intersection_area(box1, box2)
        union = box_area(box1) + box_area(box2) - intersection + 1e-6
        if box_area(box1) > 0 and box_area(box2) > 0:
            ratio1 = intersection / box_area(box1)
            ratio2 = intersection / box_area(box2)
        else:
            ratio1, ratio2 = 0, 0
        return max(intersection / union, ratio1, ratio2)

    def is_inside(box1, box2):
        # return box1[0] >= box2[0] and box1[1] >= box2[1] and box1[2] <= box2[2] and box1[3] <= box2[3]
        intersection = intersection_area(box1, box2)
        ratio1 = intersection / box_area(box1)
        return ratio1 > 0.80

    # boxes = boxes.tolist()
    filtered_boxes = []
    if ocr_bbox:
        filtered_boxes.extend(ocr_bbox)
    # print('ocr_bbox!!!', ocr_bbox)
    for i, box1_elem in enumerate(boxes):
        box1 = box1_elem['bbox']
        is_valid_box = True
        for j, box2_elem in enumerate(boxes):
            box2 = box2_elem['bbox']
            if i != j and IoU(box1, box2) > iou_threshold and box_area(box1) > box_area(box2):
                is_valid_box = False
                break
        
        if is_valid_box:
            if ocr_bbox:
                box_added = False
                ocr_labels = ''
                for box3_elem in ocr_bbox:
                    # (Hier bleibt dein Code gleich...)
                    if not box_added:
                        box3 = box3_elem['bbox']
                        if is_inside(box3, box1):
                            try:
                                ocr_labels += box3_elem['content'] + ' '
                                filtered_boxes.remove(box3_elem)
                            except: continue
                        elif is_inside(box1, box3):
                            box_added = True
                            break
                
                if not box_added:
                    if ocr_labels:
                        filtered_boxes.append({'type': 'icon', 'bbox': box1_elem['bbox'], 'interactivity': True, 'content': ocr_labels, 'source':'box_yolo_content_ocr'})
                    else:
                        filtered_boxes.append({'type': 'icon', 'bbox': box1_elem['bbox'], 'interactivity': True, 'content': None, 'source':'box_yolo_content_yolo'})
            else:
                filtered_boxes.append(box1_elem) 

    return filtered_boxes


def load_image(image_path: str) -> Tuple[np.array, torch.Tensor]:
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image_source = Image.open(image_path).convert("RGB")
    image = np.asarray(image_source)
    image_transformed, _ = transform(image_source, None)
    return image, image_transformed


def annotate(image_source: np.ndarray, boxes: torch.Tensor, logits: torch.Tensor, phrases: List[str], text_scale: float, 
             text_padding=5, text_thickness=2, thickness=3) -> np.ndarray:
    """    
    This function annotates an image with bounding boxes and labels.

    Parameters:
    image_source (np.ndarray): The source image to be annotated.
    boxes (torch.Tensor): A tensor containing bounding box coordinates. in cxcywh format, pixel scale
    logits (torch.Tensor): A tensor containing confidence scores for each bounding box.
    phrases (List[str]): A list of labels for each bounding box.
    text_scale (float): The scale of the text to be displayed. 0.8 for mobile/web, 0.3 for desktop # 0.4 for mind2web

    Returns:
    np.ndarray: The annotated image.
    """
    h, w, _ = image_source.shape
    boxes = boxes * torch.Tensor([w, h, w, h])
    xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    xywh = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xywh").numpy()
    detections = sv.Detections(xyxy=xyxy)

    labels = [f"{phrase}" for phrase in range(boxes.shape[0])]

    box_annotator = BoxAnnotator(text_scale=text_scale, text_padding=text_padding,text_thickness=text_thickness,thickness=thickness) # 0.8 for mobile/web, 0.3 for desktop # 0.4 for mind2web
    annotated_frame = image_source.copy()
    annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels, image_size=(w,h))

    label_coordinates = {f"{phrase}": v for phrase, v in zip(phrases, xywh)}
    return annotated_frame, label_coordinates


def predict(model, image, caption, box_threshold, text_threshold):
    """ Use huggingface model to replace the original model
    """
    model, processor = model['model'], model['processor']
    device = model.device

    inputs = processor(images=image, text=caption, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=box_threshold, # 0.4,
        text_threshold=text_threshold, # 0.3,
        target_sizes=[image.size[::-1]]
    )[0]
    boxes, logits, phrases = results["boxes"], results["scores"], results["labels"]
    return boxes, logits, phrases


def predict_yolo(model, image, box_threshold, imgsz, scale_img, iou_threshold=0.7):
    """ Use huggingface model to replace the original model
    """
    # model = model['model']
    if scale_img:
        result = model.predict(
        source=image,
        conf=box_threshold,
        imgsz=imgsz,
        iou=iou_threshold, # default 0.7
        )
    else:
        result = model.predict(
        source=image,
        conf=box_threshold,
        iou=iou_threshold, # default 0.7
        )
    boxes = result[0].boxes.xyxy#.tolist() # in pixel space
    conf = result[0].boxes.conf
    phrases = [str(i) for i in range(len(boxes))]

    return boxes, conf, phrases

def int_box_area(box, w, h):
    x1, y1, x2, y2 = box
    int_box = [int(x1*w), int(y1*h), int(x2*w), int(y2*h)]
    area = (int_box[2] - int_box[0]) * (int_box[3] - int_box[1])
    return area

def get_som_labeled_img(image_source: Union[str, Image.Image], model=None, BOX_TRESHOLD=0.01, output_coord_in_ratio=False, ocr_bbox=None, text_scale=0.4, text_padding=5, draw_bbox_config=None, caption_model_processor=None, ocr_text=[], use_local_semantics=True, iou_threshold=0.9,prompt=None, scale_img=False, imgsz=None, batch_size=128):
    """Process either an image path or Image object
    
    Args:
        image_source: Either a file path (str) or PIL Image object
        ...
    """
    if isinstance(image_source, str):
        image_source = Image.open(image_source)
    image_source = image_source.convert("RGB") # for CLIP
    w, h = image_source.size
    if not imgsz:
        imgsz = (h, w)
    # print('image size:', w, h)
    yolo_result = predict_yolo(model=model, image=image_source, box_threshold=BOX_TRESHOLD, imgsz=imgsz, scale_img=scale_img, iou_threshold=0.1)
    
    if yolo_result is not None and len(yolo_result) == 3:
        xyxy, logits, phrases = yolo_result
    else:
        print("WARNUNG: YOLO hat keine Icons gefunden oder gab None zurueck.")
        xyxy = torch.zeros((0, 4))
        logits = torch.zeros((0,))
        phrases = []
    xyxy = xyxy / torch.Tensor([w, h, w, h]).to(xyxy.device)
    image_source = np.asarray(image_source)
    phrases = [str(i) for i in range(len(phrases))]

    # annotate the image with labels
    if ocr_bbox:
        ocr_bbox = torch.tensor(ocr_bbox) / torch.Tensor([w, h, w, h])
        ocr_bbox=ocr_bbox.tolist()
    else:
        print('no ocr bbox!!!')
        ocr_bbox = []

    # 1. Sicherstellen, dass Listen existieren
    if ocr_bbox is None: ocr_bbox = []
    if ocr_text is None: ocr_text = []

    # 2. OCR-Elemente sicher erstellen
    ocr_bbox_elem = []
    for box, txt in zip(ocr_bbox, ocr_text):
        try:
            if int_box_area(box, w, h) > 0:
                ocr_bbox_elem.append({'type': 'text', 'bbox': box, 'interactivity': False, 'content': txt, 'source': 'box_ocr_content_ocr'})
        except: continue

    # 3. Icon-Elemente sicher erstellen
    xyxy_elem = []
    try:
        if hasattr(xyxy, 'tolist'):
            for box in xyxy.tolist():
                if int_box_area(box, w, h) > 0:
                    xyxy_elem.append({'type': 'icon', 'bbox': box, 'interactivity': True, 'content': None})
    except: pass

    # 4. �berlappung pr�fen
    filtered_boxes = remove_overlap_new(boxes=xyxy_elem, iou_threshold=iou_threshold, ocr_bbox=ocr_bbox_elem)

    # 5. Sortierung und Index-Findung (Hier lag der Fehler!)
    filtered_boxes_elem = []
    starting_idx = -1
    if filtered_boxes:
        filtered_boxes_elem = sorted(filtered_boxes, key=lambda x: x['bbox'][1])
    else:
        filtered_boxes_elem = []

    # 6. Tensor für das Modell und Zeichnen erstellen
    if filtered_boxes_elem:
        filtered_boxes_tensor = torch.tensor([box['bbox'] for box in filtered_boxes_elem])
    else:
        filtered_boxes_tensor = torch.zeros((0, 4))

    # --- Lokale Semantik (Beschreibungen für Icons generieren) ---
    if use_local_semantics and len(filtered_boxes_tensor) > 0:
        try:
            # Finde heraus, welche der sortierten Boxen Icons sind (noch keinen Content haben)
            icon_indices = [i for i, box in enumerate(filtered_boxes_elem) if box.get('content') is None]
            
            if icon_indices:
                # Nur die Icon-Boxen an das Beschreibungs-Modell schicken
                icon_boxes_tensor = torch.tensor([filtered_boxes_elem[i]['bbox'] for i in icon_indices])
                
                caption_model = caption_model_processor['model']
                if 'phi3_v' in caption_model.config.model_type: 
                    parsed_content_icon = get_parsed_content_icon_phi3v(icon_boxes_tensor, None, image_source, caption_model_processor)
                else:
                    # starting_idx ist 0, da wir den gefilterten Tensor übergeben
                    parsed_content_icon = get_parsed_content_icon(icon_boxes_tensor, 0, image_source, caption_model_processor, prompt=prompt, batch_size=batch_size)
                
                # Die generierten Texte exakt an die richtigen Stellen im sortierten Array schreiben
                for idx, caption in zip(icon_indices, parsed_content_icon):
                    filtered_boxes_elem[idx]['content'] = caption
        except Exception as e:
            print(f"Semantik-Fehler: {e}")

    # 7. Finale Listen-Zusammenführung (SYNCHRON MIT ID)
    # Jetzt ist Index i in der Liste IDENTISCH mit ID i auf dem Bild
    parsed_content_list = []
    for i, box in enumerate(filtered_boxes_elem):
        content = box.get('content', 'unknown')
        type_str = "Text Box" if box['type'] == 'text' else "Icon Box"
        parsed_content_list.append(f"{type_str} ID {i}: {content}")

    # 8. Zeichnen vorbereiten (phrases muss die gleiche Länge wie filtered_boxes_tensor haben)
    phrases = [str(i) for i in range(len(filtered_boxes_tensor))]
    
    try:
        # FALL A: Wir haben Boxen gefunden -> Zeichnen
        if len(filtered_boxes_tensor) > 0:
            phrases = [str(i) for i in range(len(filtered_boxes_tensor))]
            boxes_to_draw = box_convert(boxes=filtered_boxes_tensor, in_fmt="xyxy", out_fmt="cxcywh")

            if draw_bbox_config:
                annotated_frame, label_coordinates = annotate(image_source=image_source, boxes=boxes_to_draw, logits=logits, phrases=phrases, **draw_bbox_config)
            else:
                annotated_frame, label_coordinates = annotate(image_source=image_source, boxes=boxes_to_draw, logits=logits, phrases=phrases, text_scale=text_scale, text_padding=text_padding)
            
            pil_img = Image.fromarray(annotated_frame)
        
        # FALL B: Keine Boxen gefunden -> Originalbild nehmen
        else:
            print("INFO: Keine Boxen zum Zeichnen gefunden. Gebe Originalbild zurueck.")
            label_coordinates = {}
            pil_img = image_source if isinstance(image_source, Image.Image) else Image.fromarray(image_source)

        # Bild-Encoding (funktioniert f�r beide F�lle)
        buffered = io.BytesIO()
        pil_img.save(buffered, format="PNG")
        encoded_image = base64.b64encode(buffered.getvalue()).decode('ascii')
        
        # Koordinaten-Umrechnung nur wenn welche da sind
        if output_coord_in_ratio and len(label_coordinates) > 0:
            label_coordinates = {k: [v[0]/w, v[1]/h, v[2]/w, v[3]/h] for k, v in label_coordinates.items()}
            
        return encoded_image, label_coordinates, parsed_content_list

    except Exception as e:
        print(f"Kritischer Fehler im Zeichen-Prozess: {e}")
        # Absoluter Fallback: Leeres Resultat, damit Gradio nicht abst�rzt
        return "", {}, []


def get_xywh(input):
    x, y, w, h = input[0][0], input[0][1], input[2][0] - input[0][0], input[2][1] - input[0][1]
    x, y, w, h = int(x), int(y), int(w), int(h)
    return x, y, w, h

def get_xyxy(input):
    x, y, xp, yp = input[0][0], input[0][1], input[2][0], input[2][1]
    x, y, xp, yp = int(x), int(y), int(xp), int(yp)
    return x, y, xp, yp

def get_xywh_yolo(input):
    x, y, w, h = input[0], input[1], input[2] - input[0], input[3] - input[1]
    x, y, w, h = int(x), int(y), int(w), int(h)
    return x, y, w, h

def check_ocr_box(image_source, display_img=True, output_bb_format='xywh', goal_filtering=None, easyocr_args=None, use_paddleocr=False):
    # 1. Bild-Vorbereitung
    if isinstance(image_source, str):
        image_source = Image.open(image_source)
    if image_source.mode == 'RGBA':
        image_source = image_source.convert('RGB')
    
    w, h = image_source.size
    image_np = np.array(image_source)
    
    # Initialisierung der R�ckgabewerte
    coord = []
    text = []

    try:
        if use_paddleocr:
            # --- PADDLE OCR PFAD ---
            # Hinweis: Falls die Boxen 34, 35, 36 immer noch springen, 
            # liegt es am internen 'unwarping'. 
            result = paddle_ocr.ocr(image_np)
            
            if result and isinstance(result, list) and len(result) > 0:
                res = result[0]
                
                # VERARBEITUNG: Neues PaddleX / v3+ Format (Dictionary)
                if isinstance(res, dict) and 'rec_texts' in res and 'rec_boxes' in res:
                    texts_list = res['rec_texts']
                    boxes_list = res['rec_boxes']
                    
                    for i in range(len(texts_list)):
                        content = str(texts_list[i]).strip()
                        box = boxes_list[i]
                        
                        if len(box) >= 4:
                            # Wir berechnen die engstm�gliche Box (Tight Bounding Box)
                            # Das l�st das Problem von "zu gro�en" Boxen bei schr�gem Text
                            if isinstance(box[0], (list, np.ndarray, tuple)):
                                x_coords = [p[0] for p in box]
                                y_coords = [p[1] for p in box]
                            else:
                                # Fallback falls [x1, y1, x2, y2, ...]
                                x_coords = box[0::2]
                                y_coords = box[1::2]
                            
                            x1, y1, x2, y2 = min(x_coords), min(y_coords), max(x_coords), max(y_coords)

                            # Filter gegen Rauschen und winzige Boxen (wie Box 34-36)
                            if len(content) > 0 and (x2 - x1) > 2 and (y2 - y1) > 2:
                                coord.append([float(x1), float(y1), float(x2), float(y2)])
                                text.append(content)
                
                # VERARBEITUNG: Klassisches PaddleOCR Format (Liste)
                elif isinstance(res, list):
                    for line in res:
                        if len(line) > 1:
                            box = line[0]
                            content = line[1][0]
                            x_coords = [p[0] for p in box]
                            y_coords = [p[1] for p in box]
                            coord.append([min(x_coords), min(y_coords), max(x_coords), max(y_coords)])
                            text.append(content)
        
        else:
            # --- EASY OCR PFAD ---
            import easyocr
            reader = easyocr.Reader(['en', 'de']) 
            results = reader.readtext(image_np)
            for (bbox, txt, prob) in results:
                if prob > 0.1:
                    # Umwandlung von 4-Punkt-Polygon zu [x1, y1, x2, y2]
                    x1, y1 = bbox[0]
                    x2, y2 = bbox[2]
                    coord.append([float(x1), float(y1), float(x2), float(y2)])
                    text.append(txt)

        return (text, coord), False

    except Exception as e:
        print(f"Kritischer OCR Fehler: {e}")
        return ([], []), False

    # 4. Box-Formatierung und Visualisierung
    bb = []
    if coord: # Nur verarbeiten, wenn Boxen gefunden wurden
        if display_img:
            # Visualisierung (optional f�r Notebooks/Debug)
            opencv_img = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            for item in coord:
                x, y, a, b = get_xywh(item)
                bb.append((x, y, a, b))
                cv2.rectangle(opencv_img, (x, y), (x+a, y+b), (0, 255, 0), 2)
            plt.imshow(cv2.cvtColor(opencv_img, cv2.COLOR_BGR2RGB))
        else:
            # Standard-Formatierung f�r OmniParser
            if output_bb_format == 'xywh':
                bb = [get_xywh(item) for item in coord]
            elif output_bb_format == 'xyxy':
                bb = [get_xyxy(item) for item in coord]
    else:
        # Falls keine Boxen da sind, bleibt bb eine leere Liste
        bb = []

    # 5. Finaler Return
    return (text, bb), is_goal_filtered
