import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration, pipeline, BitsAndBytesConfig
from sentence_transformers import CrossEncoder
from PIL import Image, ImageChops, ImageEnhance
import spacy
import io
import imagehash
import base64
import warnings

warnings.filterwarnings("ignore")

device = "cuda" if torch.cuda.is_available() else "cpu"


class NewsVerifier:
    def __init__(self):
        self.fake_detector = pipeline("image-classification", model="umm-maybe/AI-image-detector", device=0 if device == "cuda" else -1)

        self.processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")

        #  4-BIT QUANTIZATION
        try:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4"
            )
            self.model = Blip2ForConditionalGeneration.from_pretrained(
                "Salesforce/blip2-opt-2.7b",
                quantization_config=bnb_config,
                device_map="auto"
            )
        except Exception as e:
            return
            self.model = Blip2ForConditionalGeneration.from_pretrained(
                "Salesforce/blip2-opt-2.7b",
                torch_dtype=torch.float16,
                device_map="auto"
            )


        self.nli_model = CrossEncoder('cross-encoder/nli-distilroberta-base', device=0)


        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            from spacy.cli import download
            download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")

        self.known_fakes_db = {
            "d4d4d4d4d4d4d4d4": "Stock Photo: 'War Zone' (2015)",
            "a1a1a1a1a1a1a1a1": "Viral Hoax: 'Shark on Highway' (2018)",
        }

    def perform_ela(self, image):
        """Generates ELA Heatmap"""
        temp_buffer = io.BytesIO()
        image.convert("RGB").save(temp_buffer, format="JPEG", quality=90)
        temp_buffer.seek(0)

        ela_image = ImageChops.difference(image.convert("RGB"), Image.open(temp_buffer))
        extrema = ela_image.getextrema()
        max_diff = max([ex[1] for ex in extrema]) or 1
        scale = 255.0 / max_diff
        ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)

        img_byte_arr = io.BytesIO()
        ela_image.save(img_byte_arr, format="PNG")
        encoded_image = base64.b64encode(img_byte_arr.getvalue()).decode('ascii')
        return f"data:image/png;base64,{encoded_image}"

    def detect_deepfake(self, image):
        try:
            results = self.fake_detector(image)
            for r in results:
                if r['label'] == 'artificial':
                    return round(r['score'] * 100, 2)
            return 0.0
        except: return 0.0

    def extract_claims(self, text):
        doc = self.nlp(text)
        entities = {"GPE": [], "DATE": [], "ORG": []}
        for ent in doc.ents:
            if ent.label_ in entities:
                entities[ent.label_].append(ent.text)
        return entities

    def ask_vqa(self, image, question):
        prompt = f"Question: {question} Answer:"
        inputs = self.processor(image, text=prompt, return_tensors="pt").to(device, torch.float16 if device == "cuda" else torch.float32)

        generated_ids = self.model.generate(**inputs, max_new_tokens=40, min_length=2, do_sample=False)
        output = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        return output.split("Answer:")[-1].strip() if "Answer:" in output else output

    def verify(self, image_bytes, text_caption):
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        flags = []

        #  HASH CHECK
        img_hash = str(imagehash.phash(image))
     
        if img_hash in self.known_fakes_db:
            flags.append({"type": "Recycled Media", "severity": "Critical", "details": f"Known fake: {self.known_fakes_db[img_hash]}"})

        # DEEPFAKE CHECK
        ai_prob = self.detect_deepfake(image)
        if ai_prob > 70:
            flags.append({"type": "⚠️ AI Generated", "severity": "Critical", "details": f"{ai_prob}% probability of being AI."})

        # GENERATE VISUAL EVIDENCE (Direct & Fast)
        # Call 1: General Scene
        visual_summary = self.ask_vqa(image, "Describe the scene, action, and location in this image.")


        # Call 2: Language
        lang_check = self.ask_vqa(image, "What language is written on the signs?")
        if "english" not in lang_check.lower() and len(lang_check) > 3:
             visual_summary += f". Text language appears to be {lang_check}."
             flags.append({"type": "Language Context", "severity": "Info", "details": f"Detected text language: {lang_check}"})

        # SEMANTIC CONSISTENCY CHECK (NLI)
        scores = self.nli_model.predict([(text_caption, visual_summary)])[0]
        contradiction_score = scores[0]
   

        label_mapping = ['contradiction', 'entailment', 'neutral']
        predicted_label = label_mapping[scores.argmax()]

        if predicted_label == 'contradiction' and contradiction_score > 0.4:
            flags.append({
                "type": "Logical Contradiction",
                "severity": "Critical",
                "details": f"Caption claims '{text_caption}', but visual evidence contradicts this (AI saw: '{visual_summary}')."
            })

        # LOCATION CHECK
        claims = self.extract_claims(text_caption)
        if claims["GPE"]:
            for loc in claims["GPE"]:
                if loc.lower() not in visual_summary.lower():
                    confirm = self.ask_vqa(image, f"Is this {loc}? Answer yes or no.")
                    if "no" in confirm.lower():
                        flags.append({
                            "type": "Location Mismatch",
                            "severity": "High",
                            "details": f"Text claims '{loc}', but visual analysis suggests otherwise."
                        })

        return {
            "ai_generated_probability": ai_prob,
            "ai_generated_caption": visual_summary,
            "detected_entities": claims,
            "inconsistencies": flags,
            "raw_vqa_checks": visual_summary,
            "ela_heatmap": self.perform_ela(image)
        }

verifier_engine = NewsVerifier()




from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import uvicorn
import os
import json


app = FastAPI()

templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
       return templates.TemplateResponse("index.html", {"request": request})


@app.post("/analyze")
async def analyze(file: UploadFile = File(...), caption: str = Form(...)):
    
    contents = await file.read()

    
    result = verifier_engine.verify(contents, caption)
    return result


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)