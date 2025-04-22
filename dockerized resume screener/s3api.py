"""
Improved API for Resume Matcher with S3/URL file download support
This module provides FastAPI endpoints for the enhanced resume matcher using boto3.
"""

from fastapi import FastAPI, Form, Query, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import time
import traceback
import json
import numpy as np
import boto3
import requests
from typing import List, Optional
from urllib.parse import urlparse
import io
from botocore.exceptions import ClientError

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

# Function to download file from S3 or HTTP URL
async def download_file(file_url: str) -> bytes:
    """
    Download file from S3 bucket or HTTP URL.
    
    Parameters:
    - file_url: The URL of the file (S3 or HTTP)
    
    Returns:
    - File content as bytes
    """
    parsed_url = urlparse(file_url)
    
    # Handle S3 URLs (s3:// protocol)
    if parsed_url.scheme == 's3':
        try:
            bucket_name = parsed_url.netloc
            object_key = parsed_url.path.lstrip('/')
            
            # Initialize S3 client
            s3_client = boto3.client('s3')
            
            # Download file from S3
            response = s3_client.get_object(Bucket=bucket_name, Key=object_key)
            return response['Body'].read()
        
        except ClientError as e:
            error_msg = f"Error downloading from S3: {str(e)}"
            print(error_msg)
            raise HTTPException(status_code=400, detail=error_msg)
    
    # Handle HTTP/HTTPS URLs
    elif parsed_url.scheme in ['http', 'https']:
        try:
            response = requests.get(file_url, timeout=30)
            response.raise_for_status()
            return response.content
        
        except requests.RequestException as e:
            error_msg = f"Error downloading from URL: {str(e)}"
            print(error_msg)
            raise HTTPException(status_code=400, detail=error_msg)
    
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported URL scheme: {parsed_url.scheme}")

# Initialize FastAPI app
app = FastAPI(
    title="Resume Matcher API",
    description="API for matching resumes against job descriptions using file URLs",
    version="3.0.0"
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
    resume_url: str = Form(..., description="URL of the resume file (S3 or HTTP)"), 
    job_description: str = Form(..., description="The job description text"),
    detailed: Optional[bool] = Query(False, description="Return detailed analysis results")
):
    """
    Endpoint to check eligibility of a resume against a job description.
    
    Parameters:
    - resume_url: The URL of the resume PDF file (S3 or HTTP)
    - job_description: The job description text
    - detailed: Whether to return detailed analysis results
    
    Returns:
    - JSON response with eligibility result
    """
    start_time = time.time()
    
    try:
        # Add logging for received URL
        print(f"API: Received resume URL: {resume_url}")
        
        # Download the PDF file
        pdf_content = await download_file(resume_url)
        print(f"API: Successfully downloaded PDF content, size: {len(pdf_content)} bytes")
        
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
    except HTTPException:
        raise  # Re-raise HTTP exceptions with original status code
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
    resume_urls: List[str] = Form(..., description="List of resume file URLs (S3 or HTTP)"),
    job_description: str = Form(..., description="The job description text")
):
    """
    Endpoint to rank multiple resumes against a job description.
    
    Parameters:
    - resume_urls: List of resume PDF file URLs (S3 or HTTP)
    - job_description: The job description text
    
    Returns:
    - JSON response with ranked resumes
    """
    start_time = time.time()
    
    try:
        # Check if URLs were provided
        if not resume_urls:
            return JSONResponse(
                content={"success": False, "error": "No resume URLs provided"},
                status_code=400
            )
        
        # Download all PDF files
        pdf_contents = []
        download_errors = []
        
        for idx, url in enumerate(resume_urls):
            try:
                content = await download_file(url)
                pdf_contents.append(content)
            except HTTPException as e:
                download_errors.append({"index": idx, "url": url, "error": str(e.detail)})
                print(f"Failed to download resume {idx}: {e.detail}")
        
        # If all downloads failed, return error
        if not pdf_contents:
            return JSONResponse(
                content={
                    "success": False, 
                    "error": "Failed to download any resume files",
                    "download_errors": download_errors
                },
                status_code=400
            )
        
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
            "num_resumes": len(resume_urls),
            "num_downloaded": len(pdf_contents),
            "num_valid": len(ranked_results),
            "processing_time": round(processing_time, 2),
            "rankings": ranked_results,
            "download_errors": download_errors if download_errors else None
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

# Configuration endpoint for AWS credentials (optional)
@app.post("/configure-aws/")
async def configure_aws(
    aws_access_key_id: str = Form(...),
    aws_secret_access_key: str = Form(...),
    aws_region: str = Form("us-east-1")
):
    """
    Configure AWS credentials for S3 access.
    This endpoint is optional if AWS credentials are set via environment variables.
    """
    try:
        # Configure boto3 with the provided credentials
        boto3.setup_default_session(
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=aws_region
        )
        
        # Test the credentials by attempting to list S3 buckets
        s3_client = boto3.client('s3')
        s3_client.list_buckets()
        
        return {"success": True, "message": "AWS credentials configured successfully"}
    except Exception as e:
        return JSONResponse(
            content={"success": False, "error": str(e)},
            status_code=400
        )

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
        "version": "3.0.0",
        "matcher_status": "initialized" if resume_matcher.model is not None else "error",
        "supports": ["S3 URLs", "HTTP URLs", "HTTPS URLs"]
    }

if __name__ == "__main__":
    # Run the FastAPI app with uvicorn when script is executed directly
    uvicorn.run(app, host="0.0.0.0", port=8000)