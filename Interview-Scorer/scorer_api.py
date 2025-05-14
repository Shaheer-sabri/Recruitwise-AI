from fastapi import FastAPI, HTTPException, Body
from typing import List, Dict, Any, Optional
import uvicorn
import os
from dotenv import load_dotenv
from interview_scorer import InterviewScorer
load_dotenv()

# Create FastAPI app
app = FastAPI(
    title="Interview Scorer API",
    description="API for evaluating interview transcripts against job descriptions",
    version="1.0.0"
)

# Create a single instance of the scorer which will be reuused for all requests instead of creating a new instance for each request
scorer = InterviewScorer(model_name="llama-3.3-70b-versatile")



# 3 Parameters that must be provided in the request body:
#      1- JD
#      2- Transcript
#      3- Skills array
@app.post("/evaluate")
async def evaluate_interview(
    job_description: str = Body(...),
    interview_transcript: Dict[str, Any] = Body(...),
    skills: List[str] = Body(...)
):
    """
    Evaluate an interview transcript against a job description for specific skills.
    
    Returns scores for each skill and an overall evaluation.
    """
    try:
        # Evaluate the interview using the scorer
        results = scorer.evaluate_interview(
            job_description=job_description,
            interview_transcript=interview_transcript,
            skills=skills
        )
        
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error evaluating interview: {str(e)}")
    

#Check if API is healthy
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "interview-scorer-api"}

# Run the API server
if __name__ == "__main__":
    # Get port from environment variable or use default
    port = int(os.getenv("PORT", 4002))
    
    # Run the server
    uvicorn.run("app:app", host="0.0.0.0", port=port)

