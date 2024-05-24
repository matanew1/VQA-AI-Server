from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from transformers import pipeline
from fastapi.middleware.cors import CORSMiddleware
import os
import tempfile
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

# Set the cache directory for Hugging Face models
os.environ['TRANSFORMERS_CACHE'] = '/tmp/huggingface_cache'

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

try:
    # Initialize the VQA pipeline
    vqa_pipe = pipeline("visual-question-answering", model="Salesforce/blip-vqa-capfilt-large", max_new_tokens=20)
    logger.info("VQA pipeline initialized successfully")
except Exception as e:
    logger.error(f"Error initializing VQA pipeline: {e}")
    raise HTTPException(status_code=500, detail="Error initializing VQA pipeline")


@app.post('/answer_question')
async def answer_question(image: UploadFile = File(...), question: str = Form(...)):
    """
    This is the VQA API
    Call this API passing an image and a question about the image.
    """
    try:
        # Save the uploaded image to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_file.write(await image.read())
            temp_file_path = temp_file.name

        # Use the VQA pipeline to get the answer
        result = vqa_pipe(image=temp_file_path, question=question)

        # Clean up the temporary file
        os.remove(temp_file_path)

        # Return the answer as JSON
        return JSONResponse(content={'answer': result[0]['answer']})

    except Exception as e:
        logger.error(f"Error processing the image and question: {e}")
        raise HTTPException(status_code=500, detail="Error processing the image and question")

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
