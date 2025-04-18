import streamlit as st
import requests
import json
import time
<<<<<<< HEAD

# Set the API URL
API_URL = "http://localhost:8000"

# Initialize session state variables
if 'session_id' not in st.session_state:
    st.session_state.session_id = None
if 'interview_started' not in st.session_state:
    st.session_state.interview_started = False
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'streaming' not in st.session_state:
    st.session_state.streaming = False

# App title
st.title("AI Interviewer Testing App")

# Sidebar for settings
with st.sidebar:
    st.header("Interview Settings")
    
    model_name = st.selectbox(
        "Model",
        ["llama3.2", "gpt-4", "claude-3"],
        index=0
    )
    
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.1)
    
    top_p = st.slider("Top P", min_value=0.0, max_value=1.0, value=0.9, step=0.1)
    
    skills = st.multiselect(
        "Skills to Test",
        ["data structures", "algorithms", "object-oriented programming", "web development", 
         "database design", "system design", "frontend", "backend", "devops"],
        default=["data structures", "algorithms", "object-oriented programming"]
    )
    
    custom_questions = st.text_area(
        "Custom Questions (one per line)",
        height=150,
        help="Enter custom questions, one per line"
    )
    
    custom_questions_list = [q.strip() for q in custom_questions.split("\n") if q.strip()]
    
    # Create a button to initialize or reset session
    if st.button("Initialize Session"):
        settings = {
            "model_name": model_name,
            "temperature": temperature,
            "top_p": top_p,
            "skills": skills,
            "custom_questions": custom_questions_list if custom_questions_list else None
        }
        
        response = requests.post(f"{API_URL}/create-session", json=settings)
        
        if response.status_code == 200:
            data = response.json()
            st.session_state.session_id = data["session_id"]
            st.session_state.interview_started = False
            st.session_state.messages = []
            st.success(f"Session created: {st.session_state.session_id}")
        else:
            st.error(f"Failed to create session: {response.text}")

    # Button to reset conversation
    if st.session_state.session_id and st.button("Reset Conversation"):
        response = requests.post(f"{API_URL}/reset/{st.session_state.session_id}")
        if response.status_code == 200:
            st.session_state.interview_started = False
            st.session_state.messages = []
            st.success("Conversation reset successfully")
        else:
            st.error(f"Failed to reset conversation: {response.text}")

# Main chat interface
if st.session_state.session_id:
    st.write(f"Session ID: {st.session_state.session_id}")
    
    # Button to start interview
    if not st.session_state.interview_started and st.button("Start Interview"):
        st.session_state.streaming = True
        
        # Make the request to start the interview
        with st.spinner("Starting interview..."):
            response = requests.post(
                f"{API_URL}/start-interview/{st.session_state.session_id}",
                stream=True
            )
            
            if response.status_code == 200:
                # Create a placeholder for the streaming text
                message_placeholder = st.empty()
                full_response = ""
                
                # Process the streaming response
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        chunk_text = chunk.decode('utf-8')
                        full_response += chunk_text
                        message_placeholder.markdown(full_response)
                
                # Add the message to the chat history
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                st.session_state.interview_started = True
                st.session_state.streaming = False
            else:
                st.error(f"Failed to start interview: {response.text}")
                st.session_state.streaming = False
    
    # Display chat messages
    st.write("### Conversation")
    chat_container = st.container()
    
    with chat_container:
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f"**You:** {message['content']}")
            else:
                st.markdown(f"**Interviewer:** {message['content']}")

    # User input
    if st.session_state.interview_started and not st.session_state.streaming:
        user_input = st.chat_input("Your response:")
        
        if user_input:
            # Add user message to chat
            st.session_state.messages.append({"role": "user", "content": user_input})
            
            # Display the new message (will be shown immediately)
            with chat_container:
                st.markdown(f"**You:** {user_input}")
            
            # Set streaming flag
            st.session_state.streaming = True
            
            # Prepare the chat message
            chat_message = {
                "message": user_input,
                "session_id": st.session_state.session_id
            }
            
            # Send message to API
            with st.spinner("Waiting for response..."):
                response = requests.post(
                    f"{API_URL}/chat/{st.session_state.session_id}",
                    json=chat_message,
                    stream=True
                )
                
                if response.status_code == 200:
                    # Create a placeholder for the streaming text
                    message_placeholder = st.empty()
                    full_response = ""
                    
                    # Process the streaming response
                    for chunk in response.iter_content(chunk_size=1024):
                        if chunk:
                            chunk_text = chunk.decode('utf-8')
                            full_response += chunk_text
                            message_placeholder.markdown(full_response)
                    
                    # Add the AI response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                    st.session_state.streaming = False
                    
                    # Rerun to update UI
                    st.rerun()
                else:
                    st.error(f"Failed to get response: {response.text}")
                    st.session_state.streaming = False
else:
    st.info("Please initialize a session using the sidebar settings")

# Display current session information at the bottom
if st.session_state.session_id:
    with st.expander("Session Information"):
        st.write(f"Current session ID: {st.session_state.session_id}")
        st.write(f"Model: {model_name}")
        st.write(f"Temperature: {temperature}")
        st.write(f"Top P: {top_p}")
        st.write(f"Skills: {', '.join(skills)}")
        if custom_questions_list:
            st.write("Custom Questions:")
            for i, q in enumerate(custom_questions_list, 1):
                st.write(f"{i}. {q}")
=======
import uuid
from typing import Dict, List, Optional
import asyncio
import websockets
import threading
import re

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
if 'username' not in st.session_state:
    st.session_state.username = ""

# Helper Functions
def create_session(settings):
    """Create a new interview session"""
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
        
        # Check if interview has ended
        if "End of interview" in full_response:
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
        st.success("Interview reset successfully!")
        return True
    else:
        st.error(f"Error resetting interview: {response.text}")
        return False

def update_settings(session_id, settings):
    """Update interview settings"""
    response = requests.post(f"{API_URL}/update-settings/{session_id}", json=settings)
    if response.status_code == 200:
        st.success("Settings updated successfully!")
        return True
    else:
        st.error(f"Error updating settings: {response.text}")
        return False

def get_conversation_history(session_id):
    """Get the full conversation history"""
    response = requests.get(f"{API_URL}/conversation-history/{session_id}")
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Error getting conversation history: {response.text}")
        return None

# UI Components
def render_setup_ui():
    """Render the setup UI for configuring the interview"""
    st.subheader("Interview Setup")
    
    with st.form("interview_settings"):
        st.markdown("### Model Settings")
        col1, col2, col3 = st.columns(3)
        with col1:
            model_name = st.text_input("Model Name", value="llama3.2")
        with col2:
            temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.1)
        with col3:
            top_p = st.slider("Top P", min_value=0.0, max_value=1.0, value=0.9, step=0.1)
        
        st.markdown("### Job Details")
        col1, col2 = st.columns(2)
        with col1:
            job_position = st.text_input("Job Position", value="entry level developer")
        with col2:
            job_description = st.text_area("Job Description", value="", height=100)
        
        st.markdown("### Questions")
        col1, col2, col3 = st.columns(3)
        with col1:
            technical_questions = st.number_input("Technical Questions", min_value=0, max_value=20, value=5)
        with col2:
            behavioral_questions = st.number_input("Behavioral Questions", min_value=0, max_value=20, value=5)
        with col3:
            max_questions = st.number_input("Max Questions", min_value=0, max_value=50, value=0, 
                                          help="0 means no limit")
        
        st.markdown("### Skills to Test")
        skills = st.text_area("Skills (one per line)", 
                              value="data structures\nalgorithms\nobject-oriented programming", 
                              height=100)
        
        st.markdown("### Custom Questions")
        custom_questions = st.text_area("Custom Questions (one per line)", 
                                      value="", 
                                      height=100)
        
        submitted = st.form_submit_button("Create Session")
        
        if submitted:
            # Parse skills and custom questions
            skills_list = [s.strip() for s in skills.split('\n') if s.strip()]
            custom_questions_list = [q.strip() for q in custom_questions.split('\n') if q.strip()]
            
            # Create settings object
            settings = {
                "model_name": model_name,
                "temperature": temperature,
                "top_p": top_p,
                "job_position": job_position,
                "job_description": job_description,
                "technical_questions": technical_questions,
                "behavioral_questions": behavioral_questions,
                "skills": skills_list,
                "custom_questions": custom_questions_list if custom_questions_list else None,
                "max_questions": max_questions if max_questions > 0 else None
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
        st.info(f"Status: {'Active' if st.session_state.interview_active else 'Not Started'}")
    with col3:
        st.info(f"Questions Asked: {st.session_state.question_count}")
    
    # Control buttons
    col1, col2, col3 = st.columns(3)
    with col1:
        if not st.session_state.interview_active:
            if st.button("Start Interview", use_container_width=True):
                start_interview(st.session_state.session_id)
                st.rerun()
        else:
            if st.button("End Interview", use_container_width=True):
                send_message(st.session_state.session_id, "end_interview", message_type="system")
                st.rerun()
    
    with col2:
        if st.button("Reset Interview", use_container_width=True):
            if reset_conversation(st.session_state.session_id):
                st.rerun()
    
    with col3:
        if st.button("Setup New Interview", use_container_width=True):
            st.session_state.session_id = None
            st.session_state.messages = []
            st.session_state.interview_active = False
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
        
        # System commands
        st.sidebar.subheader("System Commands")
        system_command = st.sidebar.text_input("Send system command")
        if st.sidebar.button("Send Command"):
            if system_command:
                response = send_message(st.session_state.session_id, system_command, message_type="system")
                st.sidebar.success(f"Command response: {response}")

# Main App
def main():
    st.set_page_config(
        page_title="AI Interviewer Tester",
        page_icon="👩‍💼",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("AI Interviewer Test Application")
    
    # Render appropriate UI based on state
    if not st.session_state.session_id:
        render_setup_ui()
    else:
        render_interview_ui()
    
    # Always render admin tools in sidebar
    render_admin_tools()

if __name__ == "__main__":
    main()
>>>>>>> 30d0e65a95a72f4a5e63445e0cdf74a30d7d4baf
