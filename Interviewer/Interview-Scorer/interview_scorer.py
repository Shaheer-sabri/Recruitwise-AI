import os
import json
from typing import List, Dict, Any
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")




class InterviewScorer:
    """
    A class that evaluates interview transcripts and provides scoring based on skills
    """
    
    def __init__(self, model_name: str = "llama-3.3-70b-versatile"):
        """Initialize the scorer with a Groq model"""
        self.model_name = model_name
        
        # Initialize the LLM if API key is available
        if GROQ_API_KEY:
            self.llm = ChatGroq(
                api_key=GROQ_API_KEY,
                model=self.model_name,
                temperature=0.2,  # Keep temperature low for more consistent scoring
                max_tokens=4000    
            )
        else:
            self.llm = None
            print("Warning: GROQ_API_KEY not found. Running in mock mode.")



    


    #Taking 3 inputs:
    # 1- JD
    # 2- Interview Transcript
    # 3- Skills array

    def evaluate_interview(self, 
                          job_description: str, 
                          interview_transcript: Dict[str, Any], 
                          skills: List[str]) -> Dict[str, Any]:
        """
        Evaluate an interview transcript and return scores
        
        Args:
            job_description: The job description text
            interview_transcript: A dictionary containing the interview transcript
            skills: A list of skills to evaluate
            
        Returns:
            A dictionary containing overall score and individual skill scores
        """
        # Get the interview history
        history = interview_transcript.get("history", [])



        
        # Validate that there's actual interview content to evaluate
        # If transcript is empty or incomplete then score is 0
        if not history:
            return {
                "error": "Empty transcript",
                "overall_score": 0,
                "skill_scores": {skill: 0 for skill in skills},
                "skill_justifications": {skill: "No interview content to evaluate" for skill in skills},
                "evaluation_summary": "Cannot evaluate an empty interview transcript. Please provide a transcript with actual interview content."
            }
        
        # Make sure there's at least one candidate response (user message)
        candidate_responses = [msg for msg in history if msg.get("role") == "user" and msg.get("content") and msg.get("content").strip() != "Let's start the interview."]
        if not candidate_responses:
            return {
                "error": "No candidate responses",
                "overall_score": 0,
                "skill_scores": {skill: 0 for skill in skills},
                "skill_justifications": {skill: "No candidate responses to evaluate" for skill in skills},
                "evaluation_summary": "The transcript doesn't contain any substantial candidate responses to evaluate."
            }
            
        # Check if we have enough content for meaningful evaluation
        if len(candidate_responses) < 3:  # Arbitrary threshold, can be adjusted
            return {
                "error": "Insufficient interview content",
                "overall_score": 0,
                "skill_scores": {skill: 0 for skill in skills},
                "skill_justifications": {skill: "Insufficient interview content" for skill in skills},
                "evaluation_summary": f"The transcript only contains {len(candidate_responses)} candidate responses, which is insufficient for a meaningful evaluation. A complete interview is needed."
            }
        



        # If we don't have an API key, generate mock results
        if not self.llm:
            return self._generate_mock_evaluation(skills)
            
        # Format the transcript for easier processing
        formatted_transcript = self._format_transcript(history)
        
        # Create the prompt for evaluation
        prompt = self._create_evaluation_prompt(job_description, formatted_transcript, skills)
        
        # Send to the model
        messages = [SystemMessage(content=prompt)]
        
        try:
            # Get response from Groq (track time but don't include in result)
            start_time = time.time()
            response = self.llm.invoke(messages)
            processing_time = time.time() - start_time
            print(f"Processing time: {processing_time:.2f} seconds")  # Only log, don't include in result
            
            # Parse the response to extract scores
            result = self._parse_scoring_response(response.content, skills)
            
            return result
        
        except Exception as e:
            print(f"Error during evaluation: {str(e)}")
            return {
                "error": str(e),
                "overall_score": 0,
                "skill_scores": {skill: 0 for skill in skills},
                "skill_justifications": {skill: "Error occurred" for skill in skills},
                "evaluation_summary": f"Failed to generate evaluation: {str(e)}"
            }
    



    # Convert from JSON to readable transcript

    def _format_transcript(self, history: List[Dict[str, str]]) -> str:
        """Format the interview history into a readable transcript"""
        formatted = ""
        
        for entry in history:
            role = entry.get("role", "")
            content = entry.get("content", "")
            
            if role == "user":
                formatted += f"Candidate: {content}\n\n"
            elif role == "assistant":
                formatted += f"Interviewer: {content}\n\n"
        
        return formatted
    


    
    # Core logic of the scorer. logic being a detailed prompt given to the scorer. 
    # Output is in JSON format created by LLM

    def _create_evaluation_prompt(self, 
                                 job_description: str, 
                                 formatted_transcript: str, 
                                 skills: List[str]) -> str:
        """Create a prompt for the LLM to evaluate the interview"""
        skills_text = ", ".join(skills)
        
        prompt = f"""
You are an expert technical interview evaluator with 20+ years of experience hiring software engineers.
You are known for your exceptionally high standards and strict technical evaluations.
Your task is to evaluate an interview transcript and provide scoring based on a job description and required skills.

# JOB DESCRIPTION:
{job_description}

# SKILLS TO EVALUATE:
{skills_text}

# INTERVIEW TRANSCRIPT:
{formatted_transcript}

# SCORING GUIDELINES:
For each skill, rigorously assess the candidate's demonstrated knowledge using this strict scoring system:
- 0: Skill was not discussed or evaluated in the interview
- 1-2: SEVERELY deficient knowledge with critical misconceptions/errors that make the candidate unsafe to hire for this skill
- 3-4: Clearly insufficient knowledge, with significant factual errors or fundamental misunderstandings
- 5-6: Basic understanding but lacks depth, with several minor conceptual errors
- 7-8: Competent understanding with good practical knowledge, minimal errors
- 9-10: Expert level knowledge with comprehensive, nuanced understanding

For the overall score, the scale means:
- 1-2: REJECT - Dangerously insufficient technical knowledge in critical areas
- 3-4: STRONG REJECT - Major gaps in fundamental concepts; not ready for this role
- 5-6: WEAK REJECT - Notable deficiencies in important areas but shows some potential
- 7-8: ACCEPTABLE - Meets expectations in most areas with minor weaknesses
- 9-10: STRONG CANDIDATE - Exceeds expectations across most or all areas

# CRITICAL INSTRUCTIONS FOR OVERALL SCORE:
- If ANY core technical skill scores 1-2, the overall score CANNOT exceed 3
- If MOST technical skills score below 4, the overall score CANNOT exceed 2
- If ALL technical skills score below 3, the overall score MUST be 1
- Strong behavioral/soft skill responses can raise a borderline score by at most 1 point
- Technical knowledge ALWAYS outweighs behavioral responses in your evaluation
- When calculating the overall score, technical accuracy is the PRIMARY consideration

# EVALUATION INSTRUCTIONS:
1. Score the candidate on each individual skill listed above on a scale from 0 to 10 using the scoring guidelines.
   - Be merciless about technical accuracy - significant errors MUST result in scores of 1-2
   - Identify all misconceptions/errors specifically in your justifications
   - Do not be lenient on technical questions regardless of how strong behavioral responses are

2. Provide an overall score on a scale from 1 to 10 following the critical instructions above.
   Consider the following with appropriate weighting:
   - Technical competence (80% of evaluation weight)
   - Problem-solving approach (15% of evaluation weight)
   - Communication and culture fit (5% of evaluation weight)

3. Provide a brief justification for each skill score (2-3 sentences per skill)
   - For skills that were not discussed (score 0), state "This skill was not covered in the interview"
   - For low scores (1-4), clearly identify the technical errors or misconceptions
   - Be brutally honest about the candidate's deficiencies

4. Provide a comprehensive evaluation summary (4-6 sentences)
   - Be direct and unsparing about technical deficiencies
   - Do not sugarcoat significant gaps in knowledge
   - Clearly state hire/no-hire recommendation based on the overall score

# OUTPUT FORMAT:
Respond in valid JSON format with the following structure:
{{
  "overall_score": <1-10 integer>,
  "skill_scores": {{
    "<skill1>": <0-10 integer>,
    "<skill2>": <0-10 integer>,
    ...
  }},
  "skill_justifications": {{
    "<skill1>": "<justification text>",
    "<skill2>": "<justification text>",
    ...
  }},
  "evaluation_summary": "<overall evaluation summary text>"
}}

IMPORTANT: Make sure your response can be parsed as valid JSON. Do not include markdown code blocks or any text outside the JSON structure.
"""
        return prompt
    



    # Extract scores from the LLMs Response

    def _parse_scoring_response(self, response_text: str, skills: List[str]) -> Dict[str, Any]:
        """Parse the LLM response to extract structured scoring data"""
        try:
            # Clean up the response text - remove any markdown code block indicators
            cleaned_text = response_text.strip()
            
            # Remove markdown code blocks if present
            if cleaned_text.startswith("```json"):
                cleaned_text = cleaned_text[7:]  # Remove ```json prefix
            if cleaned_text.startswith("```"):
                cleaned_text = cleaned_text[3:]  # Remove ``` prefix
            if cleaned_text.endswith("```"):
                cleaned_text = cleaned_text[:-3]  # Remove ``` suffix
                
            cleaned_text = cleaned_text.strip()
            
            # Try to parse the JSON response
            result = json.loads(cleaned_text)
            
            # Validate expected keys
            expected_keys = ["overall_score", "skill_scores", "evaluation_summary"]
            for key in expected_keys:
                if key not in result:
                    raise ValueError(f"Missing required key in response: {key}")
            
            # Make sure all skills are scored
            for skill in skills:
                if skill not in result["skill_scores"]:
                    # Add a default score if missing
                    result["skill_scores"][skill] = 0
            
            # Ensure skill_justifications exists
            if "skill_justifications" not in result:
                result["skill_justifications"] = {skill: "No justification provided" for skill in skills}
            
            return result
            
        except json.JSONDecodeError as e:
            # If JSON parsing fails, return an error
            print(f"Failed to parse JSON response from model: {str(e)}")
            print(f"Raw response: {response_text}")
            
            # Return a default error result
            return {
                "error": f"Failed to parse model response: {str(e)}",
                "overall_score": 0,
                "skill_scores": {skill: 0 for skill in skills},
                "skill_justifications": {skill: "Failed to parse response" for skill in skills},
                "evaluation_summary": "The model did not return a valid JSON response."
            }
        





    #IGNORE THIS
    def _generate_mock_evaluation(self, skills: List[str]) -> Dict[str, Any]:
        """Generate mock evaluation when API key is missing"""
        import random
        
        # Generate mock scores (for testing without API key)
        overall_score = random.randint(6, 9)
        skill_scores = {skill: random.randint(5, 9) for skill in skills}
        
        # Generate mock justifications
        skill_justifications = {}
        for skill in skills:
            score = skill_scores[skill]
            if score >= 8:
                justification = f"The candidate demonstrated strong knowledge of {skill} with clear explanations and practical understanding."
            elif score >= 6:
                justification = f"The candidate showed competent knowledge of {skill}, but could provide more in-depth examples."
            else:
                justification = f"The candidate has basic understanding of {skill}, but needs to develop deeper knowledge."
            
            skill_justifications[skill] = justification
        
        # Generate mock summary
        summary = "The candidate demonstrated good technical knowledge across most skills. Communication skills are strong with clear and concise explanations. They showed a proactive approach to problem-solving and teamwork. Areas for improvement include more practical experience in some technical areas."
        
        return {
            "overall_score": overall_score,
            "skill_scores": skill_scores,
            "skill_justifications": skill_justifications,
            "evaluation_summary": summary,
            "note": "MOCK EVALUATION - No API key provided"
        }
