import os
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, Depends, HTTPException, Request, BackgroundTasks
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uuid
import asyncio
from datetime import datetime, timedelta

# Import your AIInterviewer class
from model import AIInterviewer

# Models for request/response
class InterviewSettings(BaseModel):
    model_name: str = "llama3.2"
    temperature: float = 0.7
    top_p: float = 0.9
    skills: List[str] = ["data structures", "algorithms", "object-oriented programming"]
    custom_questions: Optional[List[str]] = None

class ChatMessage(BaseModel):
    message: str
    session_id: str

class SessionResponse(BaseModel):
    session_id: str
    message: str

# Session management
class SessionManager:
    def __init__(self, session_timeout_minutes: int = 30):
        self.interviewers: Dict[str, Dict[str, Any]] = {}
        self.session_timeout = timedelta(minutes=session_timeout_minutes)
    
    def create_session(self, settings: Optional[InterviewSettings] = None) -> str:
        """Create a new session with a unique ID and return the ID"""
        session_id = str(uuid.uuid4())
        
        # Create interviewer with custom settings if provided
        interviewer = AIInterviewer()
        if settings:
            interviewer.update_model_settings(
                settings.temperature, 
                settings.top_p, 
                settings.model_name
            )
            interviewer.update_skills(settings.skills)
            if settings.custom_questions:
                interviewer.update_custom_questions(settings.custom_questions)
        
        # Store interviewer and last access time
        self.interviewers[session_id] = {
            "interviewer": interviewer,
            "last_access": datetime.now()
        }
        
        return session_id
    
    def get_interviewer(self, session_id: str) -> AIInterviewer:
        """Get the interviewer for a session, updating the last access time"""
        if session_id not in self.interviewers:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Update last access time
        self.interviewers[session_id]["last_access"] = datetime.now()
        
        return self.interviewers[session_id]["interviewer"]
    
    def cleanup_expired_sessions(self):
        """Remove expired sessions"""
        current_time = datetime.now()
        expired_sessions = [
            session_id for session_id, session_data in self.interviewers.items()
            if current_time - session_data["last_access"] > self.session_timeout
        ]
        
        for session_id in expired_sessions:
            del self.interviewers[session_id]
        
        return len(expired_sessions)

# Create FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development - restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create session manager
session_manager = SessionManager()

# Background task for cleaning up expired sessions
@app.on_event("startup")
def start_cleanup_task():
    # This would ideally be an asyncio task that runs periodically
    pass  # We'll implement periodic cleanup in a production environment

# Endpoints
@app.post("/create-session", response_model=SessionResponse)
async def create_session(settings: InterviewSettings = None):
    """Create a new session with optional custom settings"""
    session_id = session_manager.create_session(settings)
    return SessionResponse(
        session_id=session_id,
        message="Session created successfully"
    )

@app.post("/start-interview/{session_id}")
async def start_interview(session_id: str):
    """Start an interview in the specified session"""
    try:
        interviewer = session_manager.get_interviewer(session_id)
        
        # Create streaming response
        async def interview_stream():
            # Convert sync generator to async
            for chunk in interviewer.start_interview():
                yield chunk
                # Small delay to simulate real-time speech pacing
                await asyncio.sleep(0.01)
        
        return StreamingResponse(interview_stream(), media_type="text/plain")
    
    except HTTPException as e:
        return {"error": e.detail}, e.status_code

@app.post("/chat/{session_id}")
async def chat(session_id: str, chat_input: ChatMessage):
    """Send a message in an existing session"""
    try:
        interviewer = session_manager.get_interviewer(session_id)
        
        # Verify the session ID in the path matches the one in the request body
        if session_id != chat_input.session_id:
            raise HTTPException(status_code=400, detail="Session ID mismatch")
        
        async def response_stream():
            # Stream the response chunks
            for chunk in interviewer.chat(chat_input.message):
                yield chunk
                # Small delay to simulate real-time speech pacing
                await asyncio.sleep(0.01)
        
        return StreamingResponse(response_stream(), media_type="text/plain")
    
    except HTTPException as e:
        return {"error": e.detail}, e.status_code

@app.post("/reset/{session_id}", response_model=SessionResponse)
def reset_conversation(session_id: str):
    """Reset the conversation for a specific session"""
    try:
        interviewer = session_manager.get_interviewer(session_id)
        result = interviewer.reset_conversation()
        return SessionResponse(
            session_id=session_id,
            message=result
        )
    
    except HTTPException as e:
        return {"error": e.detail}, e.status_code

@app.post("/update-settings/{session_id}", response_model=SessionResponse)
def update_settings(session_id: str, settings: InterviewSettings):
    """Update settings for a specific session"""
    try:
        interviewer = session_manager.get_interviewer(session_id)
        
        # Update model parameters
        result = interviewer.update_model_settings(
            settings.temperature, 
            settings.top_p, 
            settings.model_name
        )
        
        # Update skills and questions
        interviewer.update_skills(settings.skills)
        if settings.custom_questions:
            interviewer.update_custom_questions(settings.custom_questions)
        
        return SessionResponse(
            session_id=session_id,
            message=result
        )
    
    except HTTPException as e:
        return {"error": e.detail}, e.status_code

@app.get("/sessions", response_model=Dict[str, datetime])
def list_sessions():
    """List all active sessions and their last access times"""
    return {
        session_id: session_data["last_access"] 
        for session_id, session_data in session_manager.interviewers.items()
    }

@app.delete("/session/{session_id}", response_model=SessionResponse)
def delete_session(session_id: str):
    """Manually delete a session"""
    try:
        if session_id not in session_manager.interviewers:
            raise HTTPException(status_code=404, detail="Session not found")
        
        del session_manager.interviewers[session_id]
        return SessionResponse(
            session_id=session_id,
            message="Session deleted successfully"
        )
    
    except HTTPException as e:
        return {"error": e.detail}, e.status_code

# Main entry point
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)