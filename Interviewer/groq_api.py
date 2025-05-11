import os
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, Depends, HTTPException, Request, BackgroundTasks
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uuid
import asyncio
from datetime import datetime, timedelta
import json

# Import your AIInterviewer class
from model_groq import AIInterviewer

# Predefined model name that cannot be changed by users
FIXED_MODEL_NAME = "llama-3.3-70b-versatile"

# Models for request/response
class InterviewSettings(BaseModel):
    # Remove model_name from user-configurable settings
    temperature: float = 0.7
    top_p: float = 0.9
    skills: List[str] = []  # Empty array instead of placeholder
    job_position: str = ""  # Empty string instead of placeholder
    job_description: str = ""
    technical_questions: int = 5  # Default is 5
    behavioral_questions: int = 3  # Default is 3
    custom_questions: Optional[List[str]] = None
    candidate_name: str = ""  # Added candidate_name field

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
    total_expected_questions: int
    # Removed progress_percentage field

# Session management
class SessionManager:
    def __init__(self, session_timeout_minutes: int = 30):
        self.interviewers: Dict[str, Dict[str, Any]] = {}
        self.session_timeout = timedelta(minutes=session_timeout_minutes)
    
    def create_session(self, settings: Optional[InterviewSettings] = None) -> str:
        """Create a new session with a unique ID and return the ID"""
        session_id = str(uuid.uuid4())
        
        # Create interviewer with custom settings if provided, but always use the fixed model
        if settings:
            # Initialize AIInterviewer with all settings including candidate_name
            interviewer = AIInterviewer(
                model_name=FIXED_MODEL_NAME,  # Always use predefined model
                temperature=settings.temperature,
                top_p=settings.top_p,
                skills=settings.skills,
                job_position=settings.job_position,
                job_description=settings.job_description,
                technical_questions=settings.technical_questions,
                behavioral_questions=settings.behavioral_questions,
                custom_questions=settings.custom_questions,
                candidate_name=settings.candidate_name  # Pass candidate name to AIInterviewer
            )
        else:
            # Initialize with defaults and fixed model
            interviewer = AIInterviewer(model_name=FIXED_MODEL_NAME)  # Always use predefined model
        
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
    
    def save_session_to_file(self, session_id: str, output_dir: str = "interview_sessions") -> str:
        """Save a session's conversation history to a JSON file"""
        if session_id not in self.interviewers:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Get the interviewer instance
        interviewer = self.interviewers[session_id]["interviewer"]
        
        # Get session info including conversation history - removed progress_percentage
        session_info = {
            "session_id": session_id,
            "model_name": FIXED_MODEL_NAME,
            "active": interviewer.is_interview_active(),
            "questions_asked": interviewer.get_questions_asked(),
            "total_expected_questions": interviewer.get_total_expected_questions(),
            "history": interviewer.get_conversation_history()
        }
        
        # Generate a filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"interview_{session_id[:8]}_{timestamp}.json"
        filepath = os.path.join(output_dir, filename)
        
        # Write to file
        with open(filepath, 'w') as f:
            json.dump(session_info, f, indent=2)
        
        return filepath

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
    # This is where initialization is done - session_manager.create_session handles it
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
            raise HTTPException(status_code=400, detail="Interview already in progress")
        
        # Create streaming response
        async def interview_stream():
            # Convert sync generator to async
            for chunk in interviewer.start_interview():
                yield chunk
                # Small delay to simulate real-time speech pacing
                await asyncio.sleep(0.01)
        
        return StreamingResponse(interview_stream(), media_type="text/plain")
    
    except HTTPException as e:
        raise e

@app.post("/chat/{session_id}")
async def chat(session_id: str, chat_input: ChatMessage):
    """Send a message in an existing session"""
    try:
        interviewer = session_manager.get_interviewer(session_id)
        
        # Verify the session ID in the path matches the one in the request body
        if session_id != chat_input.session_id:
            raise HTTPException(status_code=400, detail="Session ID mismatch")
        
        # Check if the interview is active
        if not interviewer.is_interview_active():
            raise HTTPException(status_code=400, detail="No active interview in this session. Please start an interview first.")
        
        # Handle the message based on type (system or user)
        async def response_stream():
            # Stream the response chunks
            for chunk in interviewer.chat(chat_input.message, message_type=chat_input.type):
                yield chunk
                # Small delay to simulate real-time speech pacing
                await asyncio.sleep(0.01)
        
        return StreamingResponse(response_stream(), media_type="text/plain")
    
    except HTTPException as e:
        raise e

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
        raise e

# The update-settings endpoint code is commented out as per requirement #2
# It may be used later in production

# @app.post("/update-settings/{session_id}", response_model=SessionResponse)
# def update_settings(session_id: str, settings: InterviewSettings):
#     # Update settings for a specific session
#     try:
#         interviewer = session_manager.get_interviewer(session_id)
        
#         # Don't allow updating settings during an active interview
#         if interviewer.is_interview_active():
#             raise HTTPException(status_code=400, detail="Cannot update settings during an active interview")
        
#         messages = []
        
#         # Update job details
#         messages.append(interviewer.update_job_details(
#             settings.job_position,
#             settings.job_description
#         ))
        
#         # Update question counts
#         messages.append(interviewer.update_question_counts(
#             settings.technical_questions,
#             settings.behavioral_questions
#         ))
        
#         # Update skills
#         messages.append(interviewer.update_skills(settings.skills))
        
#         # Update custom questions if provided
#         if settings.custom_questions:
#             messages.append(interviewer.update_custom_questions(settings.custom_questions))
            
#         # Update candidate name if provided
#         if settings.candidate_name:
#             messages.append(interviewer.update_candidate_name(settings.candidate_name))
        
#         return SessionResponse(
#             session_id=session_id,
#             message="; ".join(messages)
#         )
    
#     except HTTPException as e:
#         raise e
#

@app.get("/interview-status/{session_id}", response_model=InterviewStatus)
def get_interview_status(session_id: str):
    """Check if an interview is in progress for a session and get question count"""
    try:
        interviewer = session_manager.get_interviewer(session_id)
        return InterviewStatus(
            session_id=session_id,
            active=interviewer.is_interview_active(),
            questions_asked=interviewer.get_questions_asked(),
            total_expected_questions=interviewer.get_total_expected_questions()
            # Removed progress_percentage field
        )
    
    except HTTPException as e:
        raise e

@app.get("/conversation-history/{session_id}")
def get_conversation_history(session_id: str):
    """Get the complete conversation history for evaluation purposes"""
    try:
        interviewer = session_manager.get_interviewer(session_id)
        return {
            "session_id": session_id,
            "model_name": FIXED_MODEL_NAME,  # Return fixed model name
            "active": interviewer.is_interview_active(),
            "questions_asked": interviewer.get_questions_asked(),
            "total_expected_questions": interviewer.get_total_expected_questions(),
            # Removed progress_percentage field
            "history": interviewer.get_conversation_history()
        }
    
    except HTTPException as e:
        raise e

@app.post("/save-session/{session_id}", response_model=SessionResponse)
def save_session(session_id: str, background_tasks: BackgroundTasks):
    """Save the session conversation to a file"""
    try:
        # This will run in the background so the API can respond immediately
        def save_session_background(sid):
            try:
                filepath = session_manager.save_session_to_file(sid)
                print(f"Session {sid} saved to {filepath}")
            except Exception as e:
                print(f"Error saving session {sid}: {str(e)}")
        
        # Add the task to the background tasks
        background_tasks.add_task(save_session_background, session_id)
        
        return SessionResponse(
            session_id=session_id,
            message="Session saving initiated"
        )
    
    except HTTPException as e:
        raise e

@app.get("/sessions")
def list_sessions():
    """List all active sessions and their last access times"""
    return {
        session_id: {
            "model_name": FIXED_MODEL_NAME,  # Return fixed model name
            "last_access": session_data["last_access"].isoformat(),
            "interview_active": session_data["interviewer"].is_interview_active(),
            "questions_asked": session_data["interviewer"].get_questions_asked(),
            "total_expected_questions": session_data["interviewer"].get_total_expected_questions(),
            # Removed progress_percentage field
        }
        for session_id, session_data in session_manager.interviewers.items()
    }

@app.delete("/session/{session_id}", response_model=SessionResponse)
def delete_session(session_id: str, background_tasks: BackgroundTasks):
    """Manually delete a session, saving it to a file first"""
    try:
        if session_id not in session_manager.interviewers:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Save the session before deleting it
        filepath = session_manager.save_session_to_file(session_id)
        
        # Then delete it
        del session_manager.interviewers[session_id]
        
        return SessionResponse(
            session_id=session_id,
            message=f"Session saved to {filepath} and deleted successfully"
        )
    
    except HTTPException as e:
        raise e

@app.post("/end-interview/{session_id}", response_model=SessionResponse)
async def end_interview(session_id: str):
    """Explicitly end an interview by admin command"""
    try:
        interviewer = session_manager.get_interviewer(session_id)
        
        # Check if an interview is actually in progress
        if not interviewer.is_interview_active():
            raise HTTPException(status_code=400, detail="No active interview to end")
        
        # Use the system command to end the interview
        response_content = ""
        for chunk in interviewer.chat("[SYSTEM ADMIN COMMAND: TERMINATE INTERVIEW]", message_type="system"):
            response_content += chunk
        
        return SessionResponse(
            session_id=session_id,
            message="Interview ended successfully"
        )
    
    except HTTPException as e:
        raise e

# Add an endpoint to get the model info
@app.get("/model-info")
def get_model_info():
    """Return information about the fixed model being used"""
    return {
        "model_name": FIXED_MODEL_NAME,
        "description": "Pre-configured large language model for interview simulations"
    }

# Modified security info endpoint that doesn't rely on get_cheat_attempts()
@app.get("/security-info/{session_id}")
def get_security_info(session_id: str):
    """Get information about the interview session security status"""
    try:
        interviewer = session_manager.get_interviewer(session_id)
        return {
            "session_id": session_id,
            "potential_cheat_attempts": 0,  # Hardcoded to 0 since we removed that functionality
            "active": interviewer.is_interview_active()
        }
    except HTTPException as e:
        raise e

# Main entry point
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)