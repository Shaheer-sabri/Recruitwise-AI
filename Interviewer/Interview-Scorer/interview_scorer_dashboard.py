import os
import json
import streamlit as st
import pandas as pd
from typing import List, Dict, Any
import time

# Import our InterviewScorer class
from interview_scorer import InterviewScorer

# Set page configuration
st.set_page_config(
    page_title="Interview Scorer Dashboard",
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
      "model_name": "llama-3.3-70b-versatile",
      "active": False,
      "questions_asked": 7,
      "total_expected_questions": 6,
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
        },
        {
          "role": "assistant",
          "content": "It sounds like you're eager to apply your skills and knowledge in a real-world setting and continue to grow as a software engineer. That's great to hear. \n\nBefore we dive into the technical aspects, I'd like to ask: What do you know about Khizar's Software House?"
        },
        {
          "role": "user",
          "content": "Yes, absolutely—I'm really looking forward to contributing in a meaningful way while continuing to develop as a software engineer.\n\nAs for Khizar's Software House, from what I've researched, it appears to be a growing tech company that focuses on delivering high-quality software solutions tailored to client needs. I understand that you work on a diverse range of projects, which could include web and mobile development, custom software, and perhaps even emerging technologies. What really stood out to me is the emphasis on innovation and a client-centric approach—both of which align well with my own mindset as a developer. I also get the sense that the company values collaboration and continuous improvement, which is the kind of environment where I see myself thriving and making a long-term impact."
        },
        {
          "role": "assistant",
          "content": "That's a good understanding of Khizar's Software House. It's great that you've done your research and can see how our values and approach align with your own goals and mindset.\n\nLet's move on to some technical questions. Here's the first one: Can you explain the difference between a list and a tuple in Python, and when would you use each?"
        },
        {
          "role": "user",
          "content": "Sure! In Python, both lists and tuples are used to store collections of items, but there are some key differences between them. The main difference is that lists are **mutable**, meaning you can change, add, or remove elements after the list has been created. Tuples, on the other hand, are **immutable**, so once they're created, their contents can't be modified.\n\nI would use a **list** when I need a collection of items that might change during the execution of the program—for example, if I'm dynamically adding user inputs or processing items that get updated. A **tuple** is more suitable when I want to ensure that the data remains constant, such as storing fixed pairs of values like coordinates, or returning multiple values from a function where the structure shouldn't be accidentally altered.\n\nThe immutability of tuples can also make them slightly faster and safer to use in some cases, especially when used as dictionary keys or in situations where data integrity is important."
        }
      ]
    }
    
    return sample_jd, sample_skills, json.dumps(sample_transcript, indent=2)

def display_header():
    """Display the dashboard header and instructions"""
    st.title("🎯 Interview Scorer Dashboard")
    
    with st.expander("ℹ️ About this tool", expanded=False):
        st.markdown("""
        This tool evaluates technical interview performance based on the job description, required skills, and interview transcript.
        
        ### How it works:
        1. Enter the job description for the position
        2. List the skills to evaluate (one per line)
        3. Paste the interview transcript in JSON format
        4. Click "Evaluate Interview" to get scores and feedback
        
        The scorer uses a large language model to analyze the interview and provide:
        - An overall candidate score (1-10)
        - Individual scores for each skill (1-10)
        - Justifications for each skill score
        - A summary evaluation with strengths and areas for improvement
        """)

def parse_skills(skills_input: str) -> List[str]:
    """Parse skills from a text input"""
    return [skill.strip() for skill in skills_input.split('\n') if skill.strip()]

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
    
    # Check for validation errors
    if "error" in result and result["error"] in ["Empty transcript", "No candidate responses", "Insufficient interview content"]:
        st.error(f"⚠️ Validation Error: {result['evaluation_summary']}")
        st.warning("Please provide a complete interview transcript with sufficient candidate responses.")
        return
    
    # Check for other errors
    if "error" in result:
        st.error(f"⚠️ Error: {result['error']}")
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
    
    # Check if this is mock data
    if 'note' in result and 'MOCK' in result['note']:
        st.warning("⚠️ This is mock evaluation data. Please set up your GROQ_API_KEY for real evaluations.")
    
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

def evaluate_interview(job_description: str, skills: List[str], transcript_json: str):
    """Evaluate the interview using the InterviewScorer"""
    try:
        # Parse transcript JSON
        transcript = json.loads(transcript_json)
        
        # Create scorer and evaluate
        with st.spinner("⏳ Evaluating interview... This may take a minute..."):
            scorer = InterviewScorer()
            start_time = time.time()
            result = scorer.evaluate_interview(job_description, transcript, skills)
            end_time = time.time()
            
            # Add processing time if not already included
            if "processing_time_seconds" not in result:
                result["processing_time_seconds"] = round(end_time - start_time, 2)
        
        # Check for validation errors
        if "error" in result and result["error"] in ["Empty transcript", "No candidate responses", "Insufficient interview content"]:
            st.error(f"⚠️ Validation Error: {result['evaluation_summary']}")
            
        return result
    except json.JSONDecodeError:
        st.error("❌ Invalid JSON format in the transcript. Please check and try again.")
        return {"error": "Invalid JSON format"}
    except Exception as e:
        st.error(f"❌ Error evaluating interview: {str(e)}")
        return {"error": str(e)}

def main():
    """Main function for the Streamlit dashboard"""
    # Initialize session state if needed
    if 'evaluation_result' not in st.session_state:
        st.session_state.evaluation_result = None
    
    # Display header
    display_header()
    
    # Display sidebar
    with st.sidebar:
        st.header("🛠️ Settings")
        model_name = st.selectbox(
            "Model",
            ["llama-3.3-70b-versatile", "mixtral-8x7b-32768", "gemma-7b-it"],
            index=0
        )
        
        st.caption("Note: Model selection will be implemented in a future update")
        
        st.markdown("---")
        st.subheader("📚 Resources")
        st.markdown("[GitHub Repository](https://github.com/username/interview-scorer)")
        st.markdown("[Report an Issue](https://github.com/username/interview-scorer/issues)")
    
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
            st.rerun()
        
        # Handle evaluate button
        if evaluate_button:
            if not job_description.strip():
                st.error("❌ Please enter a job description")
            elif not skills:
                st.error("❌ Please enter at least one skill to evaluate")
            elif not transcript_json.strip():
                st.error("❌ Please enter the interview transcript")
            else:
                # Save inputs to session state
                st.session_state.job_description = job_description
                st.session_state.skills_input = '\n'.join(skills)
                st.session_state.transcript_json = transcript_json
                
                # Evaluate interview
                result = evaluate_interview(job_description, skills, transcript_json)
                
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