import streamlit as st
import requests
import json
import time
import uuid
from typing import Dict, List, Optional
import asyncio

# Configuration
API_URL = "http://localhost:8000"  # Adjust if your FastAPI runs on a different host/port

# Session state initialization
if 'session_id' not in st.session_state:
    st.session_state.session_id = None
if 'interview_active' not in st.session_state:
    st.session_state.interview_active = False
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'question_count' not in st.session_state:
    st.session_state.question_count = 0
if 'total_questions' not in st.session_state:
    st.session_state.total_questions = 0
if 'candidate_name' not in st.session_state:
    st.session_state.candidate_name = ""

# Helper Functions
def create_session(settings):
    """Create a new interview session"""
    # Remove model_name from settings since it's fixed on server
    if "model_name" in settings:
        del settings["model_name"]
        
    response = requests.post(f"{API_URL}/create-session", json=settings)
    if response.status_code == 200:
        data = response.json()
        return data["session_id"]
    else:
        st.error(f"Error creating session: {response.text}")
        return None

def start_interview(session_id):
    """Start the interview"""
    response = requests.post(f"{API_URL}/start-interview/{session_id}", stream=True)
    
    if response.status_code == 200:
        st.session_state.interview_active = True
        # Add initial message placeholder
        with st.chat_message("assistant", avatar="👩‍💼"):
            message_placeholder = st.empty()
            full_response = ""
            
            # Process the streaming response
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    text_chunk = chunk.decode('utf-8')
                    full_response += text_chunk
                    message_placeholder.markdown(full_response + "▌")
            
            # Final message without cursor
            message_placeholder.markdown(full_response)
        
        # Add to message history
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        
        # Update interview status
        get_interview_status(session_id)
    else:
        st.error(f"Error starting interview: {response.text}")

def send_message(session_id, message, message_type="user"):
    """Send a message in the interview"""
    data = {
        "message": message,
        "session_id": session_id,
        "type": message_type
    }
    
    response = requests.post(f"{API_URL}/chat/{session_id}", json=data, stream=True)
    
    if response.status_code == 200:
        # Add user message to chat
        if message_type == "user":
            with st.chat_message("user", avatar="👤"):
                st.markdown(message)
            
            # Add to message history
            st.session_state.messages.append({"role": "user", "content": message})
        
        # Add assistant response with streaming effect
        with st.chat_message("assistant", avatar="👩‍💼"):
            message_placeholder = st.empty()
            full_response = ""
            
            # Process the streaming response
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    text_chunk = chunk.decode('utf-8')
                    full_response += text_chunk
                    message_placeholder.markdown(full_response + "▌")
            
            # Final message without cursor
            message_placeholder.markdown(full_response)
        
        # Add to message history
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        
        # Check if interview has ended (using the updated ending detection)
        if full_response.strip().endswith("End of interview.") or "End of interview" in full_response[-30:]:
            st.session_state.interview_active = False
            st.balloons()  # Celebrate the end of the interview!
        
        # Update interview status
        get_interview_status(session_id)
        
        return full_response
    else:
        st.error(f"Error sending message: {response.text}")
        return None

def get_interview_status(session_id):
    """Get current interview status"""
    response = requests.get(f"{API_URL}/interview-status/{session_id}")
    if response.status_code == 200:
        data = response.json()
        st.session_state.interview_active = data["active"]
        st.session_state.question_count = data["questions_asked"]
        st.session_state.total_questions = data["total_expected_questions"]
        # progress_percentage has been removed from the API
        return data
    else:
        st.error(f"Error getting interview status: {response.text}")
        return None

def reset_conversation(session_id):
    """Reset the interview"""
    response = requests.post(f"{API_URL}/reset/{session_id}")
    if response.status_code == 200:
        st.session_state.messages = []
        st.session_state.interview_active = False
        st.session_state.question_count = 0
        st.session_state.total_questions = 0
        st.success("Interview reset successfully!")
        return True
    else:
        st.error(f"Error resetting interview: {response.text}")
        return False

# Update settings function commented out as it's been removed from the API
# def update_settings(session_id, settings):
#     """Update interview settings"""
#     # Remove model_name from settings since it's fixed on server
#     if "model_name" in settings:
#         del settings["model_name"]
#         
#     response = requests.post(f"{API_URL}/update-settings/{session_id}", json=settings)
#     if response.status_code == 200:
#         st.success("Settings updated successfully!")
#         return True
#     else:
#         st.error(f"Error updating settings: {response.text}")
#         return False

def get_conversation_history(session_id):
    """Get the full conversation history"""
    response = requests.get(f"{API_URL}/conversation-history/{session_id}")
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Error getting conversation history: {response.text}")
        return None

def save_session(session_id):
    """Save the session to a file on the server"""
    response = requests.post(f"{API_URL}/save-session/{session_id}")
    if response.status_code == 200:
        st.success("Session saved successfully!")
        return True
    else:
        st.error(f"Error saving session: {response.text}")
        return False

def end_interview(session_id):
    """Explicitly end the interview"""
    response = requests.post(f"{API_URL}/end-interview/{session_id}")
    if response.status_code == 200:
        st.session_state.interview_active = False
        st.success("Interview ended successfully!")
        get_interview_status(session_id)
        return True
    else:
        st.error(f"Error ending interview: {response.text}")
        return False

def get_security_info(session_id):
    """Get security information about the session"""
    response = requests.get(f"{API_URL}/security-info/{session_id}")
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Error getting security info: {response.text}")
        return None

def get_model_info():
    """Get information about the AI model being used"""
    response = requests.get(f"{API_URL}/model-info")
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Error getting model info: {response.text}")
        return None

# UI Components
def render_setup_ui():
    """Render the setup UI for configuring the interview"""
    st.subheader("Interview Setup")
    
    # Get model info once at the beginning to display
    model_info = get_model_info()
    if model_info:
        st.info(f"Using fixed model: {model_info['model_name']}")
    
    with st.form("interview_settings"):
        st.markdown("### Model Settings")
        col1, col2 = st.columns(2)
        with col1:
            temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.1)
        with col2:
            top_p = st.slider("Top P", min_value=0.0, max_value=1.0, value=0.9, step=0.1)
        
        st.markdown("### Candidate Information")
        candidate_name = st.text_input("Candidate Name", value="", 
                                      help="Enter the candidate's name. The AI will greet them by name instead of asking.")
        
        st.markdown("### Job Details")
        col1, col2 = st.columns(2)
        with col1:
            job_position = st.text_input("Job Position", value="")
        with col2:
            job_description = st.text_area("Job Description", value="", height=100)
        
        st.markdown("### Questions")
        col1, col2 = st.columns(2)
        with col1:
            # Note: behavioral questions first, then technical
            behavioral_questions = st.number_input("Behavioral Questions", min_value=0, max_value=20, value=3)
        with col2:
            technical_questions = st.number_input("Technical Questions", min_value=0, max_value=20, value=5)
        
        st.markdown("### Skills to Test")
        skills = st.text_area("Skills (one per line)", 
                              value="", 
                              height=100)
        
        st.markdown("### Custom Questions")
        custom_questions = st.text_area("Custom Questions (one per line)", 
                                      value="", 
                                      height=100)
        
        submitted = st.form_submit_button("Create Session")
        
        if submitted:
            # Store candidate name in session state
            st.session_state.candidate_name = candidate_name
            
            # Parse skills and custom questions
            skills_list = [s.strip() for s in skills.split('\n') if s.strip()]
            custom_questions_list = [q.strip() for q in custom_questions.split('\n') if q.strip()]
            
            # Create settings object
            settings = {
                "temperature": temperature,
                "top_p": top_p,
                "job_position": job_position,
                "job_description": job_description,
                "technical_questions": technical_questions,
                "behavioral_questions": behavioral_questions,
                "skills": skills_list,
                "custom_questions": custom_questions_list if custom_questions_list else None,
                "candidate_name": candidate_name  # Add candidate name to the settings
            }
            
            # Create session
            session_id = create_session(settings)
            if session_id:
                st.session_state.session_id = session_id
                st.success(f"Session created: {session_id}")
                st.rerun()

def render_interview_ui():
    """Render the interview UI with chat interface"""
    st.subheader("AI Interview Session")
    
    # Show session info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info(f"Session ID: {st.session_state.session_id[:8]}...")
    with col2:
        status_text = "Active" if st.session_state.interview_active else "Not Started"
        if st.session_state.interview_active:
            status_text += f" - {st.session_state.question_count}/{st.session_state.total_questions} questions"
        st.info(f"Status: {status_text}")
    with col3:
        if st.session_state.candidate_name:
            st.info(f"Candidate: {st.session_state.candidate_name}")
    
    # Progress bar - Calculate progress ourselves since API no longer provides it
    if st.session_state.total_questions > 0:
        progress_percentage = min(100.0, (st.session_state.question_count / st.session_state.total_questions) * 100.0)
        progress_text = f"Interview Progress: {progress_percentage:.1f}%"
        st.progress(min(1.0, progress_percentage / 100), text=progress_text)
    
    # Control buttons
    col1, col2, col3 = st.columns(3)
    with col1:
        if not st.session_state.interview_active:
            if st.button("Start Interview", use_container_width=True):
                start_interview(st.session_state.session_id)
                st.rerun()
        else:
            if st.button("End Interview", use_container_width=True):
                end_interview(st.session_state.session_id)
                st.rerun()
    
    with col2:
        if st.button("Reset Interview", use_container_width=True):
            if reset_conversation(st.session_state.session_id):
                st.rerun()
    
    with col3:
        if st.button("Save & Start New", use_container_width=True):
            if st.session_state.session_id:
                save_session(st.session_state.session_id)
            st.session_state.session_id = None
            st.session_state.messages = []
            st.session_state.interview_active = False
            st.session_state.question_count = 0
            st.session_state.total_questions = 0
            st.session_state.candidate_name = ""
            st.rerun()
    
    # Render chat history
    st.divider()
    
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        avatar = "👩‍💼" if message["role"] == "assistant" else "👤"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])
    
    # Chat input
    if st.session_state.interview_active:
        user_input = st.chat_input("Type your response here...")
        if user_input:
            send_message(st.session_state.session_id, user_input)
            st.rerun()

def render_admin_tools():
    """Render administrative tools in the sidebar"""
    st.sidebar.subheader("Admin Controls")
    
    if st.session_state.session_id:
        # Session stats in sidebar
        st.sidebar.markdown("### Session Statistics")
        st.sidebar.info(f"Questions Asked: {st.session_state.question_count}/{st.session_state.total_questions}")
        
        # Calculate progress percentage manually
        if st.session_state.total_questions > 0:
            progress_percentage = min(100.0, (st.session_state.question_count / st.session_state.total_questions) * 100.0)
            st.sidebar.info(f"Progress: {progress_percentage:.1f}%")
        
        # Candidate info
        if st.session_state.candidate_name:
            st.sidebar.info(f"Interviewing: {st.session_state.candidate_name}")
        
        # Security info
        security_info = get_security_info(st.session_state.session_id)
        if security_info:
            st.sidebar.markdown("### Security Information")
            st.sidebar.info(f"Potential Cheat Attempts: {security_info['potential_cheat_attempts']}")
        
        # Conversation history button
        if st.sidebar.button("Get Conversation History"):
            history = get_conversation_history(st.session_state.session_id)
            if history:
                # Format for display
                formatted_history = ""
                for msg in history["history"]:
                    role = "AI" if msg["role"] == "assistant" else "User"
                    formatted_history += f"**{role}**: {msg['content']}\n\n---\n\n"
                
                st.sidebar.markdown(formatted_history)
                
                # Also offer download option
                history_json = json.dumps(history, indent=2)
                st.sidebar.download_button(
                    label="Download Conversation",
                    data=history_json,
                    file_name=f"interview_{st.session_state.session_id[:8]}.json",
                    mime="application/json"
                )
        
        # Save session button
        if st.sidebar.button("Save Session"):
            save_session(st.session_state.session_id)
        
        # System commands for admins
        st.sidebar.subheader("Administrator Commands")
        admin_password = st.sidebar.text_input("Admin Password", type="password")
        if admin_password == "admin123":  # Very simple password for testing only
            system_command = st.sidebar.text_input("Send system command")
            if st.sidebar.button("Send Command"):
                if system_command:
                    response = send_message(st.session_state.session_id, system_command, message_type="system")
                    st.sidebar.success(f"Command sent")

# Main App
def main():
    st.set_page_config(
        page_title="AI Interviewer Tester",
        page_icon="👩‍💼",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("AI Interviewer Test Application")
    
    # Add information about the app
    with st.expander("About this application"):
        st.markdown("""
        This is a testing interface for the AI Interviewer system. The AI interviewer conducts technical and behavioral interviews
        for job candidates, asking questions and following up based on their responses.
        
        **Features:**
        - Enter candidate name upfront - AI will greet them by name
        - Behavioral questions are asked first, followed by technical questions
        - The AI uses cross-questioning to probe deeper into candidate answers
        - The interview ends with a natural, professional conclusion
        - Security features prevent candidates from cheating or extracting answers
        
        **Usage:**
        1. Set up an interview with your preferred settings
        2. Enter the candidate's name to personalize the interview
        3. Start the interview and answer questions as a candidate would
        4. Admin tools are available in the sidebar for testing and monitoring
        """)
    
    # Render appropriate UI based on state
    if not st.session_state.session_id:
        render_setup_ui()
    else:
        render_interview_ui()
    
    # Always render admin tools in sidebar
    render_admin_tools()

if __name__ == "__main__":
    main()