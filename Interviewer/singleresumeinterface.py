"""
Resume Matcher Interface
A Streamlit interface for analyzing how well a resume matches a job description.
"""

import streamlit as st
import matplotlib.pyplot as plt
from singleresumematcher import ResumeMatcher

# Set page config - MUST BE THE FIRST STREAMLIT COMMAND
st.set_page_config(page_title="Resume Match Analyzer", layout="wide")

# Apply custom styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
    }
    .section-header {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    .match-card {
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 20px;
        margin: 20px 0;
        background-color: #f9f9f9;
    }
    .match-meter {
        height: 25px;
        background-color: #e0e0e0;
        border-radius: 5px;
        margin: 10px 0;
        position: relative;
    }
    .match-meter-fill {
        height: 100%;
        border-radius: 5px;
    }
    .match-meter-text {
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        color: black;
        font-weight: bold;
    }
    .match-highlight {
        font-weight: bold;
        color: #1E88E5;
        font-size: 1.5rem;
    }
    .skill-matched {
        color: black;
        background-color: white;
        border-radius: 15px;
        padding: 5px 10px;
        margin: 5px;
        display: inline-block;
    }
    .skill-missing {
        background-color: #ffebee;
        border-radius: 15px;
        padding: 5px 10px;
        margin: 5px;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# Initialize the matcher
@st.cache_resource
def get_matcher():
    return ResumeMatcher()

matcher = get_matcher()

def display_results(results):
    """Display analysis results in the Streamlit interface."""
    # Extract needed values
    final_score = results["final_score"]
    match_level = results["match_level"]
    semantic_percent = results["semantic_score"]
    skill_percent = results["skill_score"]
    resume_skills = results["resume_skills"]
    direct_matches = results["direct_matches"]
    jd_skills = results["job_skills"]
    resume_text = results["resume_text"]
    
    # Determine color based on match level
    if match_level == "High Match":
        color = "#4CAF50"  # Green
    elif match_level == "Good Match":
        color = "#FFC107"  # Yellow/Amber
    else:
        color = "#F44336"  # Red
    
    # Display results
    st.markdown("## Analysis Results")
    
    # Overall match card
    st.markdown('<div class="match-card">', unsafe_allow_html=True)
    st.markdown(f"### Overall Match: <span class='match-highlight'>{final_score}%</span> ({match_level})", unsafe_allow_html=True)
    
    # Match meter
    st.markdown(f"""
    <div class="match-meter">
        <div class="match-meter-fill" style="width: {final_score}%; background-color: {color};"></div>
        <div class="match-meter-text">{final_score}%</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Component scores
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Semantic Similarity", f"{semantic_percent}%")
    with col2:
        st.metric("Skill Match", f"{skill_percent}%")
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Skills Analysis
    st.markdown("## Skills Analysis")
    
    # Create columns for skills
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Resume Skills")
        if resume_skills:
            for skill in resume_skills:
                if skill in direct_matches:
                    st.markdown(f'<div class="skill-matched">{skill}</div>', unsafe_allow_html=True)
                else:
                    st.write(skill)
        else:
            st.write("No skills detected in the resume.")
    
    with col2:
        st.subheader("Job Description Skills")
        if jd_skills:
            for skill in jd_skills:
                is_matched = any(rs.lower() == skill.lower() for rs in resume_skills)
                if is_matched:
                    st.markdown(f'<div class="skill-matched">{skill}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="skill-missing">{skill}</div>', unsafe_allow_html=True)
        else:
            st.write("No specific skills detected in the job description.")
    
    # Visual representation of match
    st.markdown("## Similarity Breakdown")
    
    # Create a pie chart for match components
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = ['Semantic Match', 'Skill Match']
    sizes = [semantic_percent * 0.6, skill_percent * 0.4]  # Weighted contribution to final score
    colors = ['#1E88E5', '#43A047']
    explode = (0.1, 0)  # explode the 1st slice
    
    wedges, texts, autotexts = ax.pie(
        sizes, 
        explode=explode, 
        labels=labels, 
        colors=colors,
        autopct='%1.1f%%', 
        shadow=False, 
        startangle=90
    )
    for autotext in autotexts:
        autotext.set_color('white')
    
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    plt.title("Match Components (Weighted Contribution)")
    st.pyplot(fig)
    
    # Text Preview
    with st.expander("Resume Text Preview"):
        st.text_area("Extracted Text", resume_text[:2000] + "..." if len(resume_text) > 2000 else resume_text, height=200)

def main():
    """Main function to run the Streamlit app."""
    st.markdown('<p class="main-header">Resume Match Analyzer</p>', unsafe_allow_html=True)
    st.write("Upload a single resume and job description to analyze how well they match based on semantic similarity and skills.")

    # Create two columns
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown('<p class="section-header">Upload Resume</p>', unsafe_allow_html=True)
        resume_file = st.file_uploader("Upload a resume (PDF)", type=["pdf"])

    with col2:
        st.markdown('<p class="section-header">Job Description</p>', unsafe_allow_html=True)
        job_desc = st.text_area("Enter the job description:", height=250)

    analyze_button = st.button("Analyze Match", type="primary", use_container_width=True)

    if analyze_button:
        if not resume_file:
            st.warning("Please upload a resume file.")
        elif not job_desc:
            st.warning("Please enter a job description.")
        else:
            with st.spinner("Analyzing resume..."):
                # Use the matcher to analyze the resume
                results = matcher.analyze_resume(resume_file, job_desc)
                
                if not results["success"]:
                    st.error(results["error"])
                    st.stop()
                
                # Display the results
                display_results(results)
    else:
        # Initial instructions
        st.info("Upload a resume and enter a job description, then click 'Analyze Match' to see how well they align.")

if __name__ == "__main__":
    main()