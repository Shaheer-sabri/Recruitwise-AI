import os
import json
import streamlit as st
import pandas as pd
import requests
from typing import List, Dict, Any
import time

# Set page configuration
st.set_page_config(
    page_title="Interview Scorer API Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_sample_data():
    """Load sample data for testing the scorer"""
    # Sample job description
    sample_jd = """
Entry Level Software Engineer

Job Description:
We are looking for an entry-level software engineer to join our development team. The ideal candidate will have a strong foundation in computer science principles and programming skills. You will be responsible for developing and maintaining software applications, working closely with senior developers and product managers.

Key Responsibilities:
- Design, develop, and maintain software applications
- Write clean, scalable, and well-documented code
- Participate in code reviews and technical discussions
- Debug and fix issues in existing code
- Collaborate with cross-functional teams to define feature specifications

Requirements:
- Bachelor's degree in Computer Science or related field
- Proficiency in at least one programming language (Python preferred)
- Basic understanding of web development (HTML, CSS, JavaScript)
- Knowledge of database concepts and SQL
- Familiarity with RESTful APIs and web services
- Strong problem-solving skills and attention to detail
- Good communication and teamwork abilities
"""
    
    # Sample skills
    sample_skills = """Python
RESTful API
Problem Solving
Database Knowledge
Communication Skills"""
    
    # Sample transcript (simplified version of the one provided)
    sample_transcript = {
      "session_id": "9e1e2a64-1740-494d-afd7-c309af7792ea",
      "history": [
        {
          "role": "user",
          "content": "Let's start the interview."
        },
        {
          "role": "assistant",
          "content": "Hi, I am Mia, your interviewer. Welcome to your interview for the Entry Level Software Engineer position. Could you please tell me your name?"
        },
        {
          "role": "user",
          "content": "My name is zuhair "
        },
        {
          "role": "assistant",
          "content": "Nice to meet you, Zuhair. How are you today? What interests you in this Entry Level Software Engineer role, and what do you hope to achieve in this position?"
        },
        {
          "role": "user",
          "content": "I'm doing well, thank you for asking—excited to be here and have the opportunity to speak with you today. What really interests me about this Entry Level Software Engineer role is the chance to apply the skills I've built through my academic background and personal projects in a professional setting where I can continue learning and growing. I'm drawn to this position because it offers a balance between technical challenges and real-world impact, and I see it as a great opportunity to build a strong foundation in software development. I'm particularly excited about contributing to collaborative projects, learning from experienced engineers, and gradually taking on more complex responsibilities. My goal is to become a reliable team member who not only writes clean, efficient code but also contributes to the overall success of the team and the company."
        }
      ]
    }
    
    return sample_jd, sample_skills, json.dumps(sample_transcript, indent=2)

def fix_transcript_format(transcript_text_or_json: Any) -> Dict[str, Any]:
    """Fix common issues with transcript format"""
    # If it's already a dict, make a copy to avoid modifying the original
    if isinstance(transcript_text_or_json, dict):
        fixed = transcript_text_or_json.copy()
    else:
        # Try to parse as JSON if it's a string
        try:
            # Check if the string starts with a quote and not a brace
            # This handles cases where the JSON is missing the opening brace
            if isinstance(transcript_text_or_json, str):
                text = transcript_text_or_json.strip()
                if text and text[0] == '"' and not text.startswith('{'):
                    # Add the missing opening brace
                    text = '{' + text
                    # Check if it needs a closing brace
                    if not text.rstrip().endswith('}'):
                        text = text + '}'
                    
                    st.info("Fixed JSON format by adding missing braces")
                
                fixed = json.loads(text)
            else:
                return {"history": []}
                
        except json.JSONDecodeError as e:
            st.error(f"Unable to parse transcript JSON: {e}")
            st.code(transcript_text_or_json[:100] + "..." if len(str(transcript_text_or_json)) > 100 else transcript_text_or_json)
            return {"history": []}
    
    # If the transcript has session_id but no separate history field,
    # make sure we have a proper history entry
    if "session_id" in fixed and "history" in fixed:
        return fixed
    
    # If the transcript is nested inside an 'interview_transcript' field
    if "interview_transcript" in fixed and isinstance(fixed["interview_transcript"], dict):
        fixed = fixed["interview_transcript"]
    
    # If we still don't have a history field, but have a list at the top level
    if "history" not in fixed and isinstance(fixed, list):
        fixed = {"history": fixed}
    
    return fixed

def parse_skills(skills_input: str) -> List[str]:
    """Parse skills from a text input"""
    return [skill.strip() for skill in skills_input.split('\n') if skill.strip()]

def call_api_with_separate_inputs(api_url: str, job_description: str, skills: List[str], transcript_json: str):
    """Call the Interview Scorer API with separate inputs"""
    try:
        # Try to parse or fix the transcript
        try:
            # First, check if this is raw JSON or already parsed
            if isinstance(transcript_json, dict):
                transcript = transcript_json
            else:
                # Try to parse as JSON
                # Handle missing brackets by adding them if needed
                text = transcript_json.strip()
                if text and text[0] == '"' and not text.startswith('{'):
                    # Add missing opening brace
                    text = '{' + text
                    # Check if closing brace is needed
                    if not text.rstrip().endswith('}'):
                        text = text + '}'
                    st.info("Added missing braces to JSON")
                
                transcript = json.loads(text)
        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON format: {e}")
            st.code(transcript_json[:200] + "..." if len(transcript_json) > 200 else transcript_json, language="json")
            return {"error": f"Invalid JSON format: {e}"}
        
        # Fix transcript format if needed
        fixed_transcript = fix_transcript_format(transcript)
        
        # Prepare the data - sending as separate fields in the request
        data = {
            "job_description": job_description,
            "skills": skills,
            "interview_transcript": fixed_transcript
        }
        
        # Log the data structure before sending
        st.session_state.last_request_data = data
        
        # Call the API
        with st.spinner("⏳ Calling API to evaluate interview... This may take a minute..."):
            start_time = time.time()
            response = requests.post(f"{api_url}/score", json=data, timeout=120)
            end_time = time.time()
            
        # Check response
        if response.status_code == 200:
            result = response.json()
            # Add processing time
            result["processing_time_seconds"] = round(end_time - start_time, 2)
            return result
        else:
            st.error(f"API Error (Status {response.status_code})")
            st.code(response.text)
            return {
                "error": f"API returned status code {response.status_code}",
                "response": response.text
            }
    except json.JSONDecodeError:
        st.error("❌ Invalid JSON format in the transcript. Please check and try again.")
        return {"error": "Invalid JSON format in transcript"}
    except requests.exceptions.RequestException as e:
        st.error(f"❌ API Request Error: {str(e)}")
        return {"error": f"API Request Error: {str(e)}"}
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        return {"error": str(e)}

def fix_common_json_errors(json_str: str) -> str:
    """Attempt to fix common JSON format errors"""
    # Try to fix issues with missing braces
    text = json_str.strip()
    
    # Check if it starts with a quote or field name
    if text and (text[0] == '"' or text.startswith("session_id")):
        # If it doesn't start with opening brace
        if not text.startswith('{'):
            text = '{' + text
        
        # If it doesn't end with closing brace
        if not text.rstrip().endswith('}'):
            text = text + '}'
    
    # Check if we need to add quotes around field names
    for field in ["session_id", "model_name", "active", "questions_asked", "history"]:
        # Replace field: with "field":
        text = text.replace(f'{field}:', f'"{field}":')
    
    # Try parsing to verify it's fixed
    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        # If still not valid, return original
        return None

def display_input_section():
    """Display the input section of the dashboard"""
    st.header("📝 Interview Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Job Description
        st.subheader("Job Description")
        job_description = st.text_area(
            "Enter the job description",
            height=200,
            value=st.session_state.get('job_description', '')
        )
        
        # Skills
        st.subheader("Skills to Evaluate")
        skills_input = st.text_area(
            "Enter skills (one per line)",
            height=100,
            value=st.session_state.get('skills_input', '')
        )
        
        # Parse skills
        skills = parse_skills(skills_input)
        if skills:
            st.success(f"✅ {len(skills)} skills identified")
        else:
            st.warning("⚠️ No skills identified. Please enter at least one skill.")
    
    with col2:
        # Interview Transcript
        st.subheader("Interview Transcript")
        st.write("Paste the interview transcript JSON:")
        transcript_json = st.text_area(
            "Paste interview transcript JSON here",
            height=400,
            value=st.session_state.get('transcript_json', '')
        )
        
        # Auto-fix JSON option
        auto_fix_json = st.checkbox("Auto-fix JSON format issues", value=True, 
                                  help="Attempt to automatically fix common JSON format issues")
        
        # Preview of transcript if it's valid JSON
        if transcript_json:
            try:
                # Try to fix if needed
                text = transcript_json.strip()
                if auto_fix_json and text and text[0] == '"' and not text.startswith('{'):
                    # Add missing opening brace
                    text = '{' + text
                    # Check if closing brace is needed
                    if not text.rstrip().endswith('}'):
                        text = text + '}'
                    st.success("✅ Added missing braces to JSON")
                    transcript_data = json.loads(text)
                    # Update the transcript in session state
                    st.session_state.transcript_json = text
                else:
                    transcript_data = json.loads(transcript_json)
                
                # Check for history array
                if "history" in transcript_data:
                    num_exchanges = len(transcript_data["history"])
                    st.success(f"✅ Valid transcript with {num_exchanges} message exchanges")
                elif "session_id" in transcript_data and "history" in transcript_data:
                    num_exchanges = len(transcript_data["history"])
                    st.success(f"✅ Valid transcript with {num_exchanges} message exchanges")
                else:
                    st.warning("⚠️ Transcript JSON doesn't contain a 'history' array. The system will attempt to fix this.")
            except json.JSONDecodeError as e:
                st.error(f"❌ Invalid JSON format: {str(e)}")
                if auto_fix_json:
                    # Try some common fixes
                    try:
                        fixed = fix_common_json_errors(transcript_json)
                        if fixed:
                            st.success("✅ Fixed JSON format issues")
                            # Update the transcript in session state
                            st.session_state.transcript_json = fixed
                    except:
                        st.warning("Unable to automatically fix JSON format")
    
    # Action buttons
    col1, col2 = st.columns(2)
    
    with col1:
        evaluate_button = st.button("🔍 Evaluate Interview", type="primary")
    
    with col2:
        sample_button = st.button("📋 Load Sample Data")
    
    return job_description, skills, transcript_json, evaluate_button, sample_button

def display_results(result: Dict[str, Any]):
    """Display the evaluation results"""
    st.header("📊 Evaluation Results")
    
    # Check for errors
    if "error" in result:
        st.error(f"⚠️ Error: {result['error']}")
        if "response" in result:
            st.code(result["response"])
        
        # Show debugging information
        if hasattr(st.session_state, 'last_request_data'):
            with st.expander("🔍 Debug: Last Request Data"):
                st.json(st.session_state.last_request_data)
        return
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Overall score with gauge
        overall_score = result.get('overall_score', 0)
        st.metric("Overall Score", f"{overall_score}/10")
        
        # Create a progress bar for overall score
        st.progress(overall_score / 10)
        
        # Score interpretation
        if overall_score >= 8:
            st.success("Excellent candidate - Highly recommended")
        elif overall_score >= 6:
            st.info("Good candidate - Recommended with minor reservations")
        elif overall_score >= 4:
            st.warning("Average candidate - Consider additional interviews")
        else:
            st.error("Below expectations - Not recommended at this time")
    
    with col2:
        # Display summary
        st.subheader("Evaluation Summary")
        st.write(result.get('evaluation_summary', 'No summary available'))
    
    # Display skill scores
    st.subheader("Skill Scores")
    
    skill_scores = result.get('skill_scores', {})
    skill_justifications = result.get('skill_justifications', {})
    
    # Create a DataFrame for skills
    skill_data = []
    
    for skill, score in skill_scores.items():
        justification = skill_justifications.get(skill, '')
        
        # Custom status for non-evaluated skills
        status = "Not Evaluated" if score == 0 else f"{score}/10"
        
        skill_data.append({
            'Skill': skill,
            'Score': score,
            'Status': status,
            'Justification': justification
        })
    
    if skill_data:
        df = pd.DataFrame(skill_data)
        
        # Sort by score (descending) but with non-evaluated skills at the bottom
        df['SortOrder'] = df['Score'].apply(lambda x: -1 if x == 0 else x)
        df = df.sort_values('SortOrder', ascending=False).drop('SortOrder', axis=1)
        
        # Display as table
        st.dataframe(
            df,
            column_config={
                "Skill": st.column_config.TextColumn("Skill"),
                "Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=10),
                "Status": st.column_config.TextColumn("Rating"),
                "Justification": st.column_config.TextColumn("Justification")
            },
            hide_index=True
        )
        
        # Display as chart - only for evaluated skills
        evaluated_df = df[df['Score'] > 0].copy()
        if not evaluated_df.empty:
            st.subheader("Skill Score Visualization")
            chart_df = evaluated_df[['Skill', 'Score']].set_index('Skill')
            st.bar_chart(chart_df)
        else:
            st.info("No skills were evaluated in this interview.")
    
    # Processing time
    processing_time = result.get('processing_time_seconds', 0)
    st.caption(f"Processing time: {processing_time:.2f} seconds")
    
    # Offer to save results
    if st.button("💾 Save Results to File"):
        # Add timestamp
        import datetime
        result['timestamp'] = datetime.datetime.now().isoformat()
        
        # Save to file
        save_results_to_file(result)

def save_results_to_file(result: Dict[str, Any]):
    """Save results to a JSON file"""
    try:
        # Create results directory if it doesn't exist
        os.makedirs("results", exist_ok=True)
        
        # Generate filename with timestamp
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"results/interview_score_{timestamp}.json"
        
        # Write to file
        with open(filename, 'w') as f:
            json.dump(result, f, indent=2)
        
        st.success(f"✅ Results saved to {filename}")
    except Exception as e:
        st.error(f"❌ Error saving results: {str(e)}")

def check_api_health(api_url: str) -> bool:
    """Check if the API is healthy"""
    try:
        response = requests.get(f"{api_url}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def main():
    """Main function for the Streamlit dashboard"""
    # Initialize session state if needed
    if 'evaluation_result' not in st.session_state:
        st.session_state.evaluation_result = None
    
    # Display header
    st.title("🎯 Interview Scorer API Dashboard")
    st.markdown("""
    This dashboard connects to the Interview Scorer API which takes separate inputs for:
    1. Job Description
    2. Skills Array 
    3. Interview Transcript
    """)
    
    # Display sidebar
    with st.sidebar:
        st.header("🛠️ API Settings")
        
        # API URL configuration
        api_url = st.text_input(
            "API URL", 
            value=st.session_state.get('api_url', 'http://localhost:8000'),
            help="Enter the URL of the Interview Scorer API"
        )
        st.session_state.api_url = api_url
        
        # Check API health
        if st.button("Test API Connection"):
            if check_api_health(api_url):
                st.success("✅ API is available and healthy!")
            else:
                st.error("❌ Could not connect to API. Please check the URL and make sure the API is running.")
        
        st.markdown("---")
        st.markdown("### 📚 Resources")
        st.markdown("- [View API Documentation]()")
        st.markdown("- [Report an Issue]()")
    
    # Create tabs
    tab1, tab2 = st.tabs(["Input", "Results"])
    
    # Tab 1: Input Data
    with tab1:
        job_description, skills, transcript_json, evaluate_button, sample_button = display_input_section()
        
        # Handle sample data button
        if sample_button:
            sample_jd, sample_skills, sample_transcript = load_sample_data()
            st.session_state.job_description = sample_jd
            st.session_state.skills_input = sample_skills
            st.session_state.transcript_json = sample_transcript
            st.success("✅ Sample data loaded!")
            st.rerun()
        
        # Handle evaluate button
        if evaluate_button:
            if not job_description.strip():
                st.error("❌ Please enter a job description")
            elif not skills:
                st.error("❌ Please enter at least one skill to evaluate")
            elif not transcript_json.strip():
                st.error("❌ Please enter the interview transcript")
            elif not st.session_state.api_url:
                st.error("❌ Please enter the API URL in the sidebar")
            else:
                # Save inputs to session state
                st.session_state.job_description = job_description
                st.session_state.skills_input = '\n'.join(skills)
                st.session_state.transcript_json = transcript_json
                
                # Call API to evaluate interview
                result = call_api_with_separate_inputs(st.session_state.api_url, job_description, skills, transcript_json)
                
                # Store result
                st.session_state.evaluation_result = result
                
                # Switch to results tab
                st.rerun()
    
    # Tab 2: Results
    with tab2:
        if st.session_state.evaluation_result:
            display_results(st.session_state.evaluation_result)
        else:
            st.info("🔍 No evaluation results yet. Please submit an interview for evaluation.")

if __name__ == "__main__":
    main()