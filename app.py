from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from transformers import pipeline
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Initialize the VQA pipeline
vqa_pipe = pipeline("visual-question-answering", model="Salesforce/blip-vqa-capfilt-large", max_new_tokens=20)


@app.post('/answer_question')
async def answer_question(image: UploadFile = File(...), question: str = Form(...)):
    """
    This is the VQA API
    Call this api passing an image and a question about the image
    ---
    parameters:
      - name: image
        in: formData
        type: file
        required: true
      - name: question
        in: formData
        type: string
        required: true
    responses:
      200:
        description: Returns the answer to the question about the image
    """
    # Save the image locally
    image_path = 'temp_image.jpg'
    with open(image_path, 'wb') as f:
        f.write(await image.read())

    # Use the VQA pipeline to get the answer
    result = vqa_pipe(image=image_path, question=question)

    # Return the answer as JSON
    return JSONResponse(content={'answer': result[0]['answer']})


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)