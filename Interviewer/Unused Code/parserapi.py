from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Ensure you have the ResumeMatcher imported correctly
from singleresumematcher import ResumeMatcher

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the ResumeMatcher
resume_matcher = ResumeMatcher()

@app.post("/check-eligibility/")
async def check_eligibility(
    resume: UploadFile = File(...), 
    job_description: str = Form(...)
):
    """
    Endpoint to check eligibility of a resume against a job description.
    """
    try:
        # Add logging for received file
        print(f"API: Received file: {resume.filename}, content type: {resume.content_type}")
        
        # Read the uploaded PDF file
        pdf_content = await resume.read()
        print(f"API: Successfully read PDF content, size: {len(pdf_content)} bytes")
        
        # Log job description length
        print(f"API: Received job description with length: {len(job_description)} characters")
        
        # Call the get_eligibility function
        print("API: Calling ResumeMatcher.get_eligibility...")
        result = resume_matcher.get_eligibility(resume_file=pdf_content, job_description=job_description)
        print(f"API: Eligibility check complete, result: {result}")
        
        # Return the result
        return JSONResponse(content=result)
    except Exception as e:
        error_msg = f"API ERROR: Exception when processing request: {str(e)}"
        print(error_msg)
        import traceback
        print(traceback.format_exc())
        return JSONResponse(content={"success": False, "error": str(e)}, status_code=500)

# Health check endpoint
@app.get("/")
async def health_check():
    return {"status": "API is running"}