"""
Improved API for Resume Matcher with JSON serialization fixes
This module provides FastAPI endpoints for the enhanced resume matcher.
"""

from fastapi import FastAPI, File, UploadFile, Form, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import time
import traceback
import json
import numpy as np
from typing import List, Optional

# Import the improved ResumeMatcher
from singleresumematcher import ResumeMatcher

# Custom JSON encoder for NumPy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32,
                            np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

# Helper function to ensure all values are JSON serializable
def ensure_serializable(obj):
    """Recursively ensure all values in a dictionary or list are JSON serializable."""
    if isinstance(obj, dict):
        return {k: ensure_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [ensure_serializable(item) for item in obj]
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return ensure_serializable(obj.tolist())
    else:
        return obj

# Initialize FastAPI app
app = FastAPI(
    title="Resume Matcher API",
    description="API for matching resumes against job descriptions",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the ResumeMatcher with caching
resume_matcher = ResumeMatcher()

@app.post("/check-eligibility/")
async def check_eligibility(
    resume: UploadFile = File(...), 
    job_description: str = Form(...),
    detailed: Optional[bool] = Query(False, description="Return detailed analysis results")
):
    """
    Endpoint to check eligibility of a resume against a job description.
    
    Parameters:
    - resume: The resume PDF file
    - job_description: The job description text
    - detailed: Whether to return detailed analysis results
    
    Returns:
    - JSON response with eligibility result
    """
    start_time = time.time()
    
    try:
        # Add logging for received file
        print(f"API: Received file: {resume.filename}, content type: {resume.content_type}")
        
        # Read the uploaded PDF file
        pdf_content = await resume.read()
        print(f"API: Successfully read PDF content, size: {len(pdf_content)} bytes")
        
        # Log job description length
        print(f"API: Received job description with length: {len(job_description)} characters")
        
        # Call the appropriate function based on detailed flag
        print("API: Processing resume...")
        if detailed:
            result = resume_matcher.analyze_resume(resume_file=pdf_content, job_description=job_description)
            print(f"API: Detailed analysis complete")
        else:
            result = resume_matcher.get_eligibility(resume_file=pdf_content, job_description=job_description)
            print(f"API: Eligibility check complete")
        
        # Ensure result is JSON serializable
        result = ensure_serializable(result)
        
        # Add processing time to response
        processing_time = time.time() - start_time
        if isinstance(result, dict):
            result["processing_time"] = round(processing_time, 2)
        
        # Return the result with the custom encoder
        return JSONResponse(content=result, media_type="application/json")
    except Exception as e:
        error_msg = f"API ERROR: Exception when processing request: {str(e)}"
        print(error_msg)
        print(traceback.format_exc())
        
        # Return detailed error for debugging
        error_response = {
            "success": False, 
            "error": str(e),
            "traceback": traceback.format_exc(),
            "processing_time": round(time.time() - start_time, 2)
        }
        
        return JSONResponse(content=error_response, status_code=500)

@app.post("/rank-resumes/")
async def rank_resumes(
    resumes: List[UploadFile] = File(...),
    job_description: str = Form(...)
):
    """
    Endpoint to rank multiple resumes against a job description.
    
    Parameters:
    - resumes: List of resume PDF files
    - job_description: The job description text
    
    Returns:
    - JSON response with ranked resumes
    """
    start_time = time.time()
    
    try:
        # Check if files were uploaded
        if not resumes:
            return JSONResponse(
                content={"success": False, "error": "No resume files uploaded"},
                status_code=400
            )
        
        # Read all PDF files
        pdf_contents = []
        for resume in resumes:
            content = await resume.read()
            pdf_contents.append(content)
        
        # Rank the resumes
        print(f"API: Ranking {len(pdf_contents)} resumes...")
        ranked_results = resume_matcher.rank_resumes(pdf_contents, job_description)
        print(f"API: Ranking complete, found {len(ranked_results)} valid results")
        
        # Ensure results are JSON serializable
        ranked_results = ensure_serializable(ranked_results)
        
        # Add processing time and additional metadata
        processing_time = time.time() - start_time
        response = {
            "success": True,
            "num_resumes": len(pdf_contents),
            "num_valid": len(ranked_results),
            "processing_time": round(processing_time, 2),
            "rankings": ranked_results
        }
        
        return JSONResponse(content=response, media_type="application/json")
    except Exception as e:
        error_msg = f"API ERROR: Exception when ranking resumes: {str(e)}"
        print(error_msg)
        print(traceback.format_exc())
        
        error_response = {
            "success": False, 
            "error": str(e),
            "processing_time": round(time.time() - start_time, 2)
        }
        
        return JSONResponse(content=error_response, status_code=500)

@app.get("/industries/")
async def get_industries():
    """
    Get the list of supported industries for skill matching.
    """
    try:
        # Extract industry information from the matcher
        industries = list(resume_matcher.industry_skills.keys())
        return {"success": True, "industries": industries}
    except Exception as e:
        return JSONResponse(
            content={"success": False, "error": str(e)},
            status_code=500
        )

@app.get("/")
async def health_check():
    """
    Health check endpoint to verify the API is running.
    """
    return {
        "status": "API is running",
        "version": "2.0.0",
        "matcher_status": "initialized" if resume_matcher.model is not None else "error"
    }

if __name__ == "__main__":
    # Run the FastAPI app with uvicorn when script is executed directly
    uvicorn.run(app, host="0.0.0.0", port=8000)