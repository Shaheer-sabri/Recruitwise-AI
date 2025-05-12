import os
import json
from typing import Dict, List, Any, Optional, Union, Tuple
from groq import Groq
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load environment variables from .env file
load_dotenv()

class SkillScore(BaseModel):
    """Model for individual skill scores."""
    score: int = Field(..., description="Score from 1-5")
    evidence: str = Field(..., description="Evidence from the transcript")
    strengths: List[str] = Field(default_factory=list, description="Strengths demonstrated for this skill")
    weaknesses: List[str] = Field(default_factory=list, description="Areas for improvement for this skill")

class InterviewScore(BaseModel):
    """Model for the complete interview scoring."""
    skill_scores: Dict[str, SkillScore] = Field(..., description="Scores for each skill")
    overall_score: float = Field(..., description="Overall score (average of skill scores)")
    overall_assessment: str = Field(..., description="Comprehensive evaluation")
    hiring_recommendation: str = Field(..., description="Recommendation to hire or not")
    key_observations: List[str] = Field(default_factory=list, description="Key observations from the interview")

class InterviewScorer:
    """
    A scoring system for interview transcripts based on Groq's LLM models.
    
    This class evaluates candidate interview responses against required skills
    and provides detailed scoring and feedback.
    """
    
    def __init__(self, model_name: str = "llama-3.3-70b-versatile"):
        """Initialize the InterviewScorer with a specific model."""
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY environment variable is not set.")
        
        self.model_name = model_name
        self.client = Groq(api_key=self.api_key)
    
    def score_interview(
        self, 
        transcript: Union[str, Dict[str, Any]],
        skills: List[str],
        job_description: str = "",
        position_title: str = "Software Engineer"
    ) -> InterviewScore:
        """
        Score an interview based on the transcript and skills requirements.
        
        Args:
            transcript: Interview transcript (JSON string or dict)
            skills: List of skills to evaluate
            job_description: Optional job description for context
            position_title: Title of the position
            
        Returns:
            InterviewScore object with scoring details
        """
        # Process transcript if it's a string or extract history if it's a dict
        if isinstance(transcript, str):
            try:
                transcript_data = json.loads(transcript)
            except json.JSONDecodeError:
                # If it's not valid JSON, assume it's plain text conversation
                transcript_data = {"history": self._parse_plain_text(transcript)}
        else:
            transcript_data = transcript
        
        # Format the conversation for the LLM prompt
        formatted_transcript = self._format_conversation(transcript_data.get("history", []))
        
        # Generate the prompt for the LLM
        prompt = self._create_scoring_prompt(
            formatted_transcript, 
            skills, 
            job_description,
            position_title
        )
        
        # Get the evaluation from the LLM
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=2500
            )
            
            result = response.choices[0].message.content
            
            # Try to parse the result as JSON
            try:
                evaluation = json.loads(result)
                return InterviewScore(**evaluation)
            except (json.JSONDecodeError, TypeError, ValueError):
                # Fall back to structured text parsing if JSON parsing fails
                return self._parse_unstructured_response(result, skills)
                
        except Exception as e:
            raise Exception(f"Error generating score: {str(e)}")
    
    def _parse_plain_text(self, text: str) -> List[Dict[str, str]]:
        """Parse plain text conversation into a structured format."""
        lines = text.split('\n')
        messages = []
        current_role = None
        current_content = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check if this line is starting a new speaker
            if line.startswith("Interviewer:") or line.startswith("Candidate:"):
                # Save the previous message if there was one
                if current_role and current_content:
                    messages.append({
                        "role": "assistant" if current_role == "Interviewer" else "user",
                        "content": "\n".join(current_content).strip()
                    })
                    current_content = []
                
                # Set the new role
                current_role = "Interviewer" if line.startswith("Interviewer:") else "Candidate"
                # Add the content (minus the role prefix)
                current_content.append(line[len(current_role) + 1:].strip())
            else:
                # Continue with the current role
                if current_role:
                    current_content.append(line)
        
        # Add the last message
        if current_role and current_content:
            messages.append({
                "role": "assistant" if current_role == "Interviewer" else "user",
                "content": "\n".join(current_content).strip()
            })
        
        return messages
    
    def _format_conversation(self, history: List[Dict[str, str]]) -> str:
        """Format the conversation history into a readable transcript."""
        formatted = ""
        
        for message in history:
            role = message["role"]
            content = message["content"]
            
            if role == "user":
                formatted += f"Candidate: {content}\n\n"
            elif role == "assistant":
                formatted += f"Interviewer: {content}\n\n"
        
        return formatted
    
    def _create_scoring_prompt(
        self, 
        formatted_transcript: str, 
        skills: List[str],
        job_description: str,
        position_title: str
    ) -> str:
        """Create a prompt for scoring an interview."""
        
        # Add job description context if available
        job_context = f"\nJob Description:\n{job_description}\n" if job_description else ""
        
        # Join skills with better formatting
        skills_formatted = "\n".join([f"- {skill}" for skill in skills])
        
        prompt = f"""You are an expert technical interviewer and evaluator. You need to score a candidate interview for a {position_title} position based on the transcript and required skills.
{job_context}
Skills to Assess:
{skills_formatted}

Interview Transcript:
{formatted_transcript}

Scoring Criteria:
1 - No evidence of the skill or significant misconceptions
2 - Basic awareness but limited practical understanding
3 - Working knowledge with some practical application
4 - Strong understanding with clear practical experience 
5 - Expert level with deep insights and mastery

Instructions:
1. Carefully analyze the candidate's responses for evidence of each required skill.
2. Assign a score from 1-5 for each skill based on the scoring criteria.
3. Provide specific evidence from the transcript that justifies each score.
4. Identify key strengths and weaknesses for each skill (minimum 1, maximum 3 of each).
5. Calculate an overall score as the average of the individual skill scores.
6. Make a hiring recommendation based on the assessment.
7. Provide a comprehensive evaluation of the candidate.

Your evaluation must be returned in the following JSON format:
{{
    "skill_scores": {{
        "{skills[0]}": {{
            "score": X,
            "evidence": "Specific portions of the transcript that demonstrate the skill level",
            "strengths": ["Specific strength 1", "Specific strength 2"],
            "weaknesses": ["Specific weakness 1", "Specific weakness 2"]
        }},
        "{skills[1]}": {{
            "score": X,
            "evidence": "Specific portions of the transcript that demonstrate the skill level",
            "strengths": ["Specific strength 1", "Specific strength 2"],
            "weaknesses": ["Specific weakness 1", "Specific weakness 2"]
        }},
        // Repeat for all skills
    }},
    "overall_score": X.X,
    "overall_assessment": "Comprehensive evaluation of the candidate's performance",
    "hiring_recommendation": "Hire/Do Not Hire/Consider for Different Role",
    "key_observations": ["Key observation 1", "Key observation 2", "Key observation 3"]
}}

Ensure your evaluation is objective and based solely on evidence from the transcript. Be specific in your assessment and cite direct quotes or paraphrases from the candidate's responses.
        """
        
        return prompt
    
    def _parse_unstructured_response(self, text: str, skills: List[str]) -> InterviewScore:
        """Parse unstructured text response into an InterviewScore object."""
        # Default structure
        skill_scores = {}
        for skill in skills:
            skill_scores[skill] = SkillScore(
                score=0, 
                evidence="No clear evidence found",
                strengths=["No specific strengths identified"],
                weaknesses=["Insufficient information to assess"]
            )
        
        overall_score = 0.0
        overall_assessment = "Unable to parse a structured assessment from the model output."
        hiring_recommendation = "Unable to determine"
        key_observations = ["Response format error", "Manual review recommended"]
        
        # Try to extract information from the text
        lines = text.strip().split('\n')
        current_skill = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check for skill headers
            for skill in skills:
                if skill in line and ":" in line:
                    current_skill = skill
                    # Try to extract score
                    for i in range(1, 6):
                        if f"score: {i}" in line.lower() or f"score:{i}" in line.lower() or f"score: {i}/" in line.lower():
                            skill_scores[skill].score = i
                            break
            
            # If we're in a skill section, look for evidence
            if current_skill and "evidence" in line.lower() and ":" in line:
                parts = line.split(":", 1)
                if len(parts) > 1:
                    skill_scores[current_skill].evidence = parts[1].strip()
            
            # Look for strengths
            if current_skill and ("strength" in line.lower() or "pro" in line.lower()) and ":" in line:
                parts = line.split(":", 1)
                if len(parts) > 1:
                    strengths_text = parts[1].strip()
                    # Try to split by commas, bullets, or other common separators
                    strengths = [s.strip().strip('•-*') for s in re.split(r'[,;•\n-]', strengths_text) if s.strip()]
                    if strengths:
                        skill_scores[current_skill].strengths = strengths
            
            # Look for weaknesses
            if current_skill and ("weakness" in line.lower() or "con" in line.lower() or "improvement" in line.lower()) and ":" in line:
                parts = line.split(":", 1)
                if len(parts) > 1:
                    weaknesses_text = parts[1].strip()
                    # Try to split by commas, bullets, or other common separators
                    weaknesses = [w.strip().strip('•-*') for w in re.split(r'[,;•\n-]', weaknesses_text) if w.strip()]
                    if weaknesses:
                        skill_scores[current_skill].weaknesses = weaknesses
            
            # Look for overall assessment
            if "overall assessment" in line.lower() and ":" in line:
                parts = line.split(":", 1)
                if len(parts) > 1:
                    overall_assessment = parts[1].strip()
            
            # Look for hiring recommendation
            if "recommendation" in line.lower() and ":" in line:
                parts = line.split(":", 1)
                if len(parts) > 1:
                    hiring_recommendation = parts[1].strip()
            
            # Look for overall score
            if "overall score" in line.lower() and ":" in line:
                parts = line.split(":", 1)
                if len(parts) > 1:
                    try:
                        overall_score = float(parts[1].strip().split()[0])
                    except (ValueError, IndexError):
                        pass
        
        # If we couldn't extract an overall score, calculate it
        if overall_score == 0.0:
            valid_scores = [s.score for s in skill_scores.values() if s.score > 0]
            if valid_scores:
                overall_score = sum(valid_scores) / len(valid_scores)
        
        # Create the InterviewScore object
        return InterviewScore(
            skill_scores=skill_scores,
            overall_score=overall_score,
            overall_assessment=overall_assessment,
            hiring_recommendation=hiring_recommendation,
            key_observations=key_observations
        )