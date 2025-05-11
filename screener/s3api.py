from fastapi import FastAPI, File, UploadFile, Form, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import boto3
import io
import time
import traceback
from typing import Optional

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

# Initialize S3 client
s3_client = boto3.client('s3')

# Helper function to get resume content from URL
async def get_resume_from_url(resume_url: str):
    # Parse the S3 URL
    if resume_url.startswith('s3://'):
        parts = resume_url.replace('s3://', '').split('/', 1)
        bucket_name = parts[0]
        object_key = parts[1]
        
        try:
            print(f"API: Attempting to download file from S3: {bucket_name}/{object_key}")
            response = s3_client.get_object(Bucket=bucket_name, Key=object_key)
            pdf_content = response['Body'].read()
            print(f"API: Successfully downloaded PDF from S3, size: {len(pdf_content)} bytes")
            source = "s3"
        except Exception as s3_error:
            error_msg = f"Failed to download file from S3: {str(s3_error)}"
            print(f"API ERROR: {error_msg}")
            raise Exception(error_msg)
    else:
        # Support for HTTP URLs as well
        import requests
        try:
            response = requests.get(resume_url, timeout=30)
            response.raise_for_status()
            pdf_content = response.content
            print(f"API: Successfully downloaded PDF from HTTP URL, size: {len(pdf_content)} bytes")
            source = "url"
        except Exception as url_error:
            error_msg = f"Failed to download file from URL: {str(url_error)}"
            raise Exception(error_msg)
    
    return pdf_content, source

@app.post("/check-eligibility/")
async def check_eligibility(
    resume_url: str = Form(...), 
    job_description: str = Form(...)
):
    """
    Endpoint to check basic eligibility of a resume against a job description.
    """
    try:
        # Get the resume content
        pdf_content, source = await get_resume_from_url(resume_url)
        
        # Log job description length
        print(f"API: Received job description with length: {len(job_description)} characters")
        
        # Process the resume for basic eligibility
        print("API: Processing resume for basic eligibility...")
        full_result = resume_matcher.get_eligibility(resume_file=pdf_content, job_description=job_description)
        print(f"API: Eligibility check complete")
        
        # Create simplified response with only success and score
        result = {
            "success": full_result.get("success", False),
            "score": full_result.get("score", 0)
        }
        
        # Return the result
        return JSONResponse(content=result, media_type="application/json")
    except Exception as e:
        error_msg = f"API ERROR: Exception when processing request: {str(e)}"
        print(error_msg)
        print(traceback.format_exc())
        
        # Return error response
        error_response = {
            "success": False, 
            "error": str(e),
            "traceback": traceback.format_exc()
        }
        
        return JSONResponse(content=error_response, status_code=500)

@app.post("/analyze-resume/")
async def analyze_resume(
    resume_url: str = Form(...), 
    job_description: str = Form(...)
):
    """
    Endpoint to get detailed analysis of a resume against a job description.
    """
    try:
        # Get the resume content
        pdf_content, source = await get_resume_from_url(resume_url)
        
        # Log job description length
        print(f"API: Received job description with length: {len(job_description)} characters")
        
        # Process the resume for detailed analysis
        print("API: Processing resume for detailed analysis...")
        analysis_result = resume_matcher.analyze_resume(resume_file=pdf_content, job_description=job_description)
        print(f"API: Detailed analysis complete")
        
        # Add source to the result dictionary
        if isinstance(analysis_result, dict):
            analysis_result["source"] = source
            
        # Return the detailed result
        return JSONResponse(content=analysis_result, media_type="application/json")
    except Exception as e:
        error_msg = f"API ERROR: Exception when processing request: {str(e)}"
        print(error_msg)
        print(traceback.format_exc())
        
        # Return detailed error for debugging
        error_response = {
            "success": False, 
            "error": str(e),
            "traceback": traceback.format_exc(),
        }
        
        return JSONResponse(content=error_response, status_code=500)

# Health check endpoint
@app.get("/")
async def health_check():
    return {"status": "API is running", "matcher_status": "initialized" if resume_matcher.model is not None else "error"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)