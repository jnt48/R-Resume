from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
import base64
import io
from PIL import Image
import PyPDF2
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

def get_gemini_response(job_description_text: str, pdf_text: str, prompt: str) -> str:
    """
    Calls the Gemini model with the job description, extracted PDF text,
    and any additional prompt instructions.
    """
    model = genai.GenerativeModel('gemini-1.5-flash')
    # We pass the text content instead of an image
    response = model.generate_content([job_description_text, pdf_text, prompt])
    return response.text

def extract_pdf_text(uploaded_file: bytes) -> str:
    """
    Extracts text from the uploaded PDF using PyPDF2.
    """
    try:
        # Read PDF from bytes
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file))
        text_content = []
        for page in pdf_reader.pages:
            text_content.append(page.extract_text() or "")
        return "\n".join(text_content)
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
        # Extract text from the uploaded PDF
        pdf_text = extract_pdf_text(await uploaded_file.read())
        
        # Determine which prompt to use
        if mode == "evaluation":
            prompt = """
            You are an experienced Technical Human Resource Manager. Review the provided resume text against the job description.
            Share your professional evaluation on whether the candidate's profile aligns with the role.
            Highlight the strengths and weaknesses in relation to the specified job requirements.
            """
        elif mode == "match_percentage":
            prompt = """
            You are a skilled ATS scanner. Evaluate the resume text against the job description.
            Provide a percentage match, followed by missing keywords and final thoughts.
            """
        else:
            raise HTTPException(status_code=400, detail="Invalid mode. Use 'evaluation' or 'match_percentage'.")

        # Generate a response from the Gemini model
        response_text = get_gemini_response(job_description, pdf_text, prompt)
        return {"response": response_text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
