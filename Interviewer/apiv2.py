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
from modelv2 import AIInterviewer

# Models for request/response
class InterviewSettings(BaseModel):
    model_name: str = "llama3.2"
    temperature: float = 0.7
    top_p: float = 0.9
    skills: List[str] = ["computer science"]
    job_position: str = "entry level developer"
    job_description: str = ""
    technical_questions: int = 5
    behavioral_questions: int = 5
    max_questions: Optional[int] = None
    custom_questions: Optional[List[str]] = None

class ChatMessage(BaseModel):
    message: str
    session_id: str
    type: str = "user"  # Options: "user" (candidate) or "system" (admin commands)

class SessionResponse(BaseModel):
    session_id: str
    message: str

class InterviewStatus(BaseModel):
    session_id: str
    active: bool
    questions_asked: int
    max_questions: Optional[int] = None

# Session management
class SessionManager:
    def __init__(self, session_timeout_minutes: int = 30):
        self.interviewers: Dict[str, Dict[str, Any]] = {}
        self.session_timeout = timedelta(minutes=session_timeout_minutes)
    
    def create_session(self, settings: Optional[InterviewSettings] = None) -> str:
        """Create a new session with a unique ID and return the ID"""
        session_id = str(uuid.uuid4())
        
        # Create interviewer with custom settings if provided
        if settings:
            interviewer = AIInterviewer(
                model_name=settings.model_name,
                temperature=settings.temperature,
                top_p=settings.top_p,
                skills=settings.skills,
                job_position=settings.job_position,
                job_description=settings.job_description,
                technical_questions=settings.technical_questions,
                behavioral_questions=settings.behavioral_questions,
                max_questions=settings.max_questions,
                custom_questions=settings.custom_questions
            )
        else:
            interviewer = AIInterviewer()
        
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
async def start_cleanup_task():
    """Start a background task to clean up expired sessions periodically"""
    async def cleanup_periodically():
        while True:
            try:
                removed = session_manager.cleanup_expired_sessions()
                if removed > 0:
                    print(f"Cleaned up {removed} expired sessions")
            except Exception as e:
                print(f"Error in cleanup task: {str(e)}")
            # Run every 5 minutes
            await asyncio.sleep(300)
    
    # Start the background task
    asyncio.create_task(cleanup_periodically())

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
        
        # Check if an interview is already in progress
        if interviewer.is_interview_active():
            return {"error": "Interview already in progress"}, 400
        
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
        
        # Handle the message based on type (system or user)
        async def response_stream():
            # Stream the response chunks
            for chunk in interviewer.chat(chat_input.message, message_type=chat_input.type):
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
        messages = []
        
        # Update model parameters
        messages.append(interviewer.update_model_settings(
            settings.temperature, 
            settings.top_p, 
            settings.model_name
        ))
        
        # Update job details
        messages.append(interviewer.update_job_details(
            settings.job_position,
            settings.job_description
        ))
        
        # Update question counts including max_questions
        messages.append(interviewer.update_question_counts(
            settings.technical_questions,
            settings.behavioral_questions,
            settings.max_questions
        ))
        
        # Update skills
        messages.append(interviewer.update_skills(settings.skills))
        
        # Update custom questions if provided
        if settings.custom_questions:
            messages.append(interviewer.update_custom_questions(settings.custom_questions))
        
        return SessionResponse(
            session_id=session_id,
            message="; ".join(messages)
        )
    
    except HTTPException as e:
        return {"error": e.detail}, e.status_code

@app.get("/interview-status/{session_id}", response_model=InterviewStatus)
def get_interview_status(session_id: str):
    """Check if an interview is in progress for a session and get question count"""
    try:
        interviewer = session_manager.get_interviewer(session_id)
        return InterviewStatus(
            session_id=session_id,
            active=interviewer.is_interview_active(),
            questions_asked=interviewer.get_questions_asked(),
            max_questions=interviewer.max_questions
        )
    
    except HTTPException as e:
        return {"error": e.detail}, e.status_code

@app.get("/conversation-history/{session_id}")
def get_conversation_history(session_id: str):
    """Get the complete conversation history for evaluation purposes"""
    try:
        interviewer = session_manager.get_interviewer(session_id)
        return {
            "session_id": session_id,
            "active": interviewer.is_interview_active(),
            "questions_asked": interviewer.get_questions_asked(),
            "history": interviewer.get_conversation_history()
        }
    
    except HTTPException as e:
        return {"error": e.detail}, e.status_code

@app.get("/sessions")
def list_sessions():
    """List all active sessions and their last access times"""
    return {
        session_id: {
            "last_access": session_data["last_access"],
            "interview_active": session_data["interviewer"].is_interview_active(),
            "questions_asked": session_data["interviewer"].get_questions_asked(),
            "max_questions": session_data["interviewer"].max_questions
        }
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
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)