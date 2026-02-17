import cv2
import base64
import httpx
import time

# 1. Setup Camera
cam = cv2.VideoCapture(0)

def capture_and_ask(prompt):
    ret, frame = cam.read()
    if not ret:
        return "Camera failed"

    # Encode image to base64 for the LLM
    _, buffer = cv2.imencode('.jpg', frame)
    image_base64 = base64.b64encode(buffer).decode('utf-8')

    # 2. Call local Ollama (Assumes you have: ollama run llama3.2-vision)
    # Or change to 'minicpm-v' if you have that pulled
    response = httpx.post('http://localhost:11434/api/generate', 
        json={
            "model": "llama3.2-vision",
            "prompt": prompt,
            "images": [image_base64],
            "stream": False
        }, timeout=30.0)
    
    return response.json().get('response')

# 3. The "Rioc" Loop
print("Rioc is watching...")
persona_prompt = "You are Rioc, King of the Dead. Look at this image. Describe the person you see and decide if they are a threat. Speak in a haunting, ancient tone."

try:
    while True:
        result = capture_and_ask(persona_prompt)
        print(f"\n[RIOC]: {result}")
        time.sleep(5) # Wait 5 seconds before looking again
except KeyboardInterrupt:
    cam.release()
