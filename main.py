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
    df = pd.read_csv('prompts_with_outfit_type.csv', dtype={'id': str})
    PROMPT_DATABASE = pd.Series(df.prompt.values, index=df.id).to_dict()
    PROMPT_DATABASE_customer_try_on = pd.Series(df.customerTryOn.values, index=df.id).to_dict()
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

        if not response.candidates:
            print("🚨 Error: Gemini API returned no candidates. The prompt may have been blocked.")
            if hasattr(response, 'prompt_feedback'):
                print(f"   Prompt Feedback: {response.prompt_feedback}")
            return None

        # Check for the image data in the response
        for part in response.candidates[0].content.parts:
            if hasattr(part, 'inline_data') and part.inline_data:
                print("✅ Image data received from Gemini.")
                return part.inline_data.data # Return the image data if found

        # If the loop finishes, no image was found. Print debug info.
        print("🚨 Error: Gemini API did not return an image.")
        print(f"   Finish Reason: {response.candidates[0].finish_reason}")
        print(f"   Safety Ratings: {response.candidates[0].safety_ratings}")
        try:
            print(f"   Text Response: {response.text}")
        except Exception as e:
            print(f"   Could not retrieve text response: {e}")

        return None # Return None because no image was found

    except Exception as e:
        print(f"🚨 An unexpected error occurred during image generation: {e}")
        return None


def generate_customer_try_on(fabric_image_bytes, customer_image_bytes, text_prompt):
    if not text_prompt:
        print("🚨 Error: No prompt provided.")
        return None

    try:
        print("✅ Preparing images and prompt for Gemini customer try-on...")
        fabric_swatch_image = PIL.Image.open(BytesIO(fabric_image_bytes))
        customer_image = PIL.Image.open(BytesIO(customer_image_bytes))

        model = genai.GenerativeModel('gemini-2.5-flash-image')

        print("🤖 Sending request to Gemini API for customer try-on...")
        response = model.generate_content([text_prompt, fabric_swatch_image, customer_image])

        if not response.candidates:
            print("🚨 Error: Gemini API returned no candidates for customer try-on.")
            # Check prompt feedback for blocking information
            if hasattr(response, 'prompt_feedback'):
                print(f"   Prompt Feedback: {response.prompt_feedback}")
                if hasattr(response.prompt_feedback, 'block_reason'):
                    print(f"   Block Reason: {response.prompt_feedback.block_reason}")
                else:
                    print(f"   Block Reason: N/A")
                if hasattr(response.prompt_feedback, 'safety_ratings'):
                    print(f"   Prompt Safety Ratings: {response.prompt_feedback.safety_ratings}")
            return None

        # Check for the image data in the response
        for part in response.candidates[0].content.parts:
            if hasattr(part, 'inline_data') and part.inline_data:
                print("✅ Image data received from Gemini for customer try-on.")
                return part.inline_data.data # Return the image data if found

        # If the loop finishes, no image was found. Print debug info.
        print("🚨 Error: Gemini API did not return an image for customer try-on.")
        print(f"   Finish Reason: {response.candidates[0].finish_reason}")
        print(f"   Safety Ratings: {response.candidates[0].safety_ratings}") 
        # Try to get finish message if available
        if hasattr(response.candidates[0], 'finish_message'):
            print(f"   Finish Message: {response.candidates[0].finish_message}")
        
        try:
            print(f"   Text Response: {response.text}")
        except Exception as e:
            print(f"   Could not retrieve text response: {e}")

        return None # Return None because no image was found

    except Exception as e:
        print(f"🚨 An unexpected error occurred during customer try-on image generation: {e}")
        import traceback
        print(f"   Full traceback: {traceback.format_exc()}")
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
        df = pd.read_csv('prompts.csv', dtype={'id': str})
        outfit_list = df[['id', 'outfit']].to_dict(orient='records')

        return outfit_list
    except Exception as e:
        print(f"🚨 Error reading prompts CSV for GET request: {e}")
        return {"error": "Could not load outfit list."}

@app.post("/generate_mockup/")
async def create_mockup(prompt_id: str = Form(...), fabric_file: UploadFile = File(...)):
    print(f"Received request for prompt_id: {prompt_id}")
    prompt_to_use = PROMPT_DATABASE.get(prompt_id)

    if not prompt_to_use:
        return {"error": f"Prompt ID '{prompt_id}' not found in prompts database."}

    fabric_bytes = await fabric_file.read()
    image_data = generate_mannequin_mockup(fabric_bytes, prompt_to_use)

    if image_data:
        return StreamingResponse(BytesIO(image_data), media_type="image/png")
    else:
        return {"error": "Could not generate image"}


PROMPT_TEMPLATE_CUSTOMER_TRY_ON = """Inputs

Image 1: Fabric swatch (for pattern, texture, color)

Image 2: Person and background (for pose, body, and scene)

Task: Photorealistic Virtual Try-On

Design Outfit: [Outfit_type]


Apply to Image 2:

Replace the person's original clothing (upper and lower) with the new outfit.

Fit the outfit realistically to the person's body and pose, creating natural folds.

Map the Image 1 fabric pattern and texture naturally onto the outfit, respecting seams and scale.

Seamlessly match all lighting, highlights, and shadows from Image 2.

Strict Constraints:

Generate only the described outfit (no new accessories).

Preserve the person's face, pose, and body identically.

Preserve the background of Image 2 identically."""


@app.post("/customer_try_on/")
async def create_customer_try_on(prompt_id: str = Form(...), fabric_file: UploadFile = File(...), customer_file: UploadFile = File(...)):
    print(f"Received request for customer try-on with prompt_id: {prompt_id}")
    outfit_type = PROMPT_DATABASE_customer_try_on.get(prompt_id)

    if not outfit_type:
        return {"error": f"Outfit type for prompt ID '{prompt_id}' not found in prompts database."}

    prompt_to_use = PROMPT_TEMPLATE_CUSTOMER_TRY_ON.replace("[Outfit_type]", outfit_type)

    fabric_bytes = await fabric_file.read()
    customer_bytes = await customer_file.read()
    image_data = generate_customer_try_on(fabric_bytes, customer_bytes, prompt_to_use)

    if image_data:
        return StreamingResponse(BytesIO(image_data), media_type="image/png")
    else:
        return {"error": "Could not generate image for customer try-on"}


print("✅ FastAPI app and endpoint are defined.")