import google.generativeai as genai
import PIL.Image
from io import BytesIO
import pandas as pd
import os # Import the os module to get environment variables
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware # Import CORS Middleware

# --- 1. CONFIGURE API KEY FROM ENVIRONMENT VARIABLE ---
GEMINI_API_KEY = 'AIzaSyB2X2AcVXUVikzK4KQYpNLCb2sgFwTfNW0'
genai.configure(api_key=GEMINI_API_KEY)

# --- 2. LOAD PROMPTS ---
PROMPT_DATABASE = {}
try:
    df = pd.read_csv('prompts.csv', dtype={'id': str})
    PROMPT_DATABASE = pd.Series(df.prompt.values, index=df.id).to_dict()
    print("✅ Prompts loaded successfully.")
except FileNotFoundError:
    print("🚨 FATAL ERROR: prompts.csv not found.")

# --- 3. CORE LOGIC ---
def generate_mannequin_mockup(fabric_image_bytes, text_prompt):
    if not text_prompt:
        print("🚨 Error: No prompt provided.")
        return None

    try:
        print("✅ Preparing images and prompt for Gemini...")
        fabric_swatch_image = PIL.Image.open(BytesIO(fabric_image_bytes))

        # IMPORTANT: Make sure this model name is correct for image generation
        model = genai.GenerativeModel('gemini-2.5-flash-image')

        print("🤖 Sending request to Gemini API...")
        response = model.generate_content([text_prompt, fabric_swatch_image])

        # Check for the image data in the response
        for part in response.candidates[0].content.parts:
            if hasattr(part, 'inline_data') and part.inline_data:
                print("✅ Image data received from Gemini.")
                return part.inline_data.data # Return the image data if found

        # If the loop finishes, no image was found. Print debug info.
        print("🚨 Error: Gemini API did not return an image.")
        if response.candidates[0].finish_reason.name != "STOP":
             print(f"   Finish Reason: {response.candidates[0].finish_reason.name}")
             print(f"   Safety Ratings: {response.candidates[0].safety_ratings}")
        if response.text:
            print(f"   Text Response: {response.text}")

        return None # Return None because no image was found

    except Exception as e:
        print(f"🚨 An unexpected error occurred during image generation: {e}")
        return None

# --- 4. CREATE THE FASTAPI APP ---
app = FastAPI()

# Add CORS Middleware to allow requests from your Lovable app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/prompts/")
async def get_prompts():
    try:
        df = pd.read_csv('/content/prompts.csv', dtype={'id': str})
        outfit_list = df[['id', 'outfit']].to_dict(orient='records')

        return outfit_list
    except Exception as e:
        print(f"🚨 Error reading prompts CSV for GET request: {e}")
        return {"error": "Could not load outfit list."}

@app.post("/generate_mockup/")
async def create_mockup(prompt_id: str = Form(...), file: UploadFile = File(...)):
    print(f"Received request for prompt_id: {prompt_id}")
    prompt_to_use = PROMPT_DATABASE.get(prompt_id)

    if not prompt_to_use:
        return {"error": f"Prompt ID '{prompt_id}' not found in prompts database."}

    fabric_bytes = await file.read()
    image_data = generate_mannequin_mockup(fabric_bytes, prompt_to_use)

    if image_data:
        return StreamingResponse(BytesIO(image_data), media_type="image/png")
    else:
        return {"error": "Could not generate image"}

print("✅ FastAPI app and endpoint are defined.")