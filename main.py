from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
import base64
import io
from PIL import Image
import pdf2image
import google.generativeai as genai

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

app = FastAPI(title="ATS Resume Expert API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_gemini_response(input_text, pdf_content, prompt):
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content([input_text, pdf_content[0], prompt])
    return response.text

def input_pdf_setup(uploaded_file: bytes):
    try:
        images = pdf2image.convert_from_bytes(uploaded_file)
        first_page = images[0]
        img_byte_arr = io.BytesIO()
        first_page.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()
        pdf_parts = [{
            "mime_type": "image/jpeg",
            "data": base64.b64encode(img_byte_arr).decode()
        }]
        return pdf_parts
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing PDF: {e}")

@app.post("/analyze_resume")
async def analyze_resume(
    job_description: str = Form(...),
    uploaded_file: UploadFile = File(...),
    mode: str = Form(...)
):
    if not uploaded_file:
        raise HTTPException(status_code=400, detail="No file uploaded.")
    
    try:
        pdf_content = input_pdf_setup(await uploaded_file.read())
        
        if mode == "evaluation":
            prompt = """
            You are an experienced Technical Human Resource Manager. Review the provided resume against the job description.
            Share your professional evaluation on whether the candidate's profile aligns with the role.
            Highlight the strengths and weaknesses in relation to the specified job requirements.
            """
        elif mode == "match_percentage":
            prompt = """
            You are a skilled ATS scanner. Evaluate the resume against the job description.
            Provide a percentage match, followed by missing keywords and final thoughts.
            """
        else:
            raise HTTPException(status_code=400, detail="Invalid mode. Use 'evaluation' or 'match_percentage'.")
        
        response = get_gemini_response(job_description, pdf_content, prompt)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
