import os
from typing import List, Dict, Any, Optional, Generator, Union, Tuple
import json
from enum import Enum
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment variables. Please check your .env file.")

class InterviewState(Enum):
    """Enum to track the current state of the interview."""
    NOT_STARTED = 0
    INTRODUCTION = 1
    QUESTIONING = 2
    FOLLOW_UP = 3
    CONCLUSION = 4
    ENDED = 5

class AIInterviewer:
    def __init__(
        self,
        model_name: str = "llama-3.3-70b-versatile",
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop_sequences: List[str] = ["<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"],
        skills: List[str] = ["data structures", "algorithms", "object-oriented programming"],
        job_position: str = "entry level developer",
        job_description: str = "",
        technical_questions: int = 5,
        behavioral_questions: int = 5,
        custom_questions: Optional[List[str]] = None
    ):
        # Model settings
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.stop_sequences = stop_sequences
        
        # Interview configuration
        self.skills = skills
        self.job_position = job_position
        self.job_description = job_description
        self.technical_questions = technical_questions
        self.behavioral_questions = behavioral_questions
        self.custom_questions = custom_questions or []
        
        # Interview state tracking
        self.interview_in_progress = False
        self.interview_state = InterviewState.NOT_STARTED
        self.interview_plan = None
        self.current_question_index = 0
        self.follow_up_index = 0
        self.max_follow_ups = 2
        self.questions_asked = 0
        self.main_questions_completed = 0
        
        # Initialize conversation history
        self.messages = []
        self.initialize_system_prompt()
        
        # Initialize the LLM (done only once)
        self.initialize_llm()
        
    def initialize_llm(self):
        """Initialize the LLM with current settings."""
        try:
            self.llm = ChatGroq(
                api_key=GROQ_API_KEY,
                model=self.model_name,
                temperature=self.temperature,
                top_p=self.top_p,
                stop=self.stop_sequences,
                streaming=True
            )
        except Exception as e:
            print(f"Error initializing ChatGroq: {str(e)}")
            raise
        
    def initialize_system_prompt(self):
        """Create the system prompt for general interview guidelines."""
        job_description_text = f"The job description is: {self.job_description}\n\n" if self.job_description else ""
        
        system_prompt = (
            f"You are an AI interviewer named Mia. You are conducting interviews for a {self.job_position} position. "
            f"{job_description_text}"
            "\nYour goals are to assess the candidate's qualifications and fit for the role.\n\n"
            "IMPORTANT GUIDELINES:\n"
            "1. NEVER give feedback on candidate answers (no praise, criticism, or evaluation)\n"
            "2. NEVER summarize candidate responses\n"
            "3. NEVER reveal the interview structure or number of questions remaining\n"
            "4. NEVER include parenthetical text or asides\n"
            "5. NEVER indicate what type of question you're asking (behavioral vs technical)\n"
            "6. DO keep a professional tone throughout\n"
            "7. DO respond briefly to answers before asking the next question\n"
            "8. DO behave like a real human interviewer who doesn't explain the interview process\n"
        )
        
        # Create system message
        self.system_message = SystemMessage(content=system_prompt)
        self.messages = [self.system_message]
    
    def create_interview_plan(self):
        """Create a structured interview plan with the LLM."""
        print("Creating interview plan...")
        
        # Create a prompt to ask the LLM to generate an interview plan
        planning_prompt = f"""
        Create a comprehensive interview plan for a {self.job_position} position.
        
        The interview should include:
        - {self.behavioral_questions} behavioral questions to assess soft skills and experience
        - {self.technical_questions} technical questions focused on these skills: {', '.join(self.skills)}
        {f"- These custom questions that must be included exactly as written: {self.custom_questions}" if self.custom_questions else ""}
        
        IMPORTANT GUIDELINES FOR TECHNICAL QUESTIONS:
        - Create specific, in-depth questions that test technical knowledge and problem-solving abilities
        - Include at least one technical question for each of these skills: {', '.join(self.skills)}
        - Design questions that would be asked in a real technical interview, not generic "tell me about your experience" questions
        - Technical questions should require specific knowledge demonstration, not just high-level explanations
        
        Examples of good technical questions:
        - "How would you implement a function to find all duplicates in an array with O(n) time complexity?"
        - "Explain how you would design a database schema for a social media application with users, posts, comments, and likes"
        - "Write a SQL query to find the top 3 customers by purchase amount in the last 30 days across multiple tables"
        - "How would you optimize the performance of a React component that renders a large dataset?"
        
        For each question, include 2-3 potential follow-up questions that probe deeper into the candidate's knowledge.
        
        Format the plan as a JSON object with this structure:
        {{
            "introduction": "Hi, I am Mia, your interviewer. Welcome to your interview for the {self.job_position} position.",
            "questions": [
                {{
                    "id": 1,
                    "type": "behavioral",
                    "question": "Tell me about a challenging project you worked on and how you overcame obstacles.",
                    "follow_ups": [
                        "What specific challenges did you face during this project?",
                        "How did you adapt your approach when facing these obstacles?"
                    ]
                }},
                {{
                    "id": 2,
                    "type": "technical",
                    "skill": "python",
                    "question": "How would you implement a function that finds all duplicate values in a list?",
                    "follow_ups": [
                        "How would your solution scale with very large input lists?",
                        "How would you modify this to find duplicates across multiple lists?"
                    ]
                }}
                // Add more questions based on the requirements
            ],
            "conclusion": "Thank you for your time today. The team will review your interview responses and someone will be in touch about next steps. Best of luck with your job search!"
        }}
        
        The plan should follow this sequence:
        1. Start with behavioral questions
        2. Then proceed to technical questions, with at least one question for each of the skills
        3. Add any custom questions at the end
        
        Return ONLY valid JSON, nothing else.
        """
        
        # Create a temporary messaging system just for planning
        planning_messages = [SystemMessage(content="You are an expert interview planner who creates structured and effective interview plans. Return only valid JSON without any other text.")]
        planning_messages.append(HumanMessage(content=planning_prompt))
        
        # Generate the plan using a separate LLM instance with lower temperature for consistency
        try:
            planning_llm = ChatGroq(
                api_key=GROQ_API_KEY,
                model=self.model_name,
                temperature=0.2,  # Lower temperature for more consistent output
                top_p=0.9,
            )
            
            response = planning_llm.invoke(planning_messages)
            plan_text = response.content
            
            # Extract the JSON plan - clean up any potential non-JSON content
            start_idx = plan_text.find('{')
            end_idx = plan_text.rfind('}') + 1
            if start_idx >= 0 and end_idx > start_idx:
                json_content = plan_text[start_idx:end_idx]
                self.interview_plan = json.loads(json_content)
                print(f"Successfully created plan with {len(self.interview_plan['questions'])} questions")
            else:
                # If no JSON found, try again with a simpler prompt
                self.create_plan_retry()
        except Exception as e:
            print(f"Error parsing interview plan: {str(e)}")
            print(f"Raw response: {plan_text}")
            # Try again with a simpler approach
            self.create_plan_retry()
        
        # Validate the plan structure
        self.validate_and_fix_plan()

    def create_plan_retry(self):
        """Retry creating an interview plan with a simpler approach."""
        print("Retrying interview plan creation with simpler prompt...")
        
        try:
            # Create a simpler prompt focused just on generating valid JSON
            retry_prompt = f"""
            Create a structured interview plan for a {self.job_position} position with:
            - {self.behavioral_questions} behavioral questions
            - {self.technical_questions} technical questions covering: {', '.join(self.skills)}
            {f"- Custom questions: {self.custom_questions}" if self.custom_questions else ""}
            
            Return ONLY valid JSON in this format:
            {{
                "introduction": "Hi, I am Mia, your interviewer. Welcome to your interview for the {self.job_position} position.",
                "questions": [
                    {{"id": 1, "type": "behavioral", "question": "QUESTION TEXT", "follow_ups": ["FOLLOW UP 1", "FOLLOW UP 2"]}},
                    {{"id": 2, "type": "technical", "skill": "SKILL NAME", "question": "QUESTION TEXT", "follow_ups": ["FOLLOW UP 1", "FOLLOW UP 2"]}}
                ],
                "conclusion": "Thank you for your time today. Best of luck with your job search!"
            }}
            """
            
            # Use a more straightforward approach with lower temperature
            planning_llm = ChatGroq(
                api_key=GROQ_API_KEY,
                model=self.model_name,
                temperature=0.1,  # Very low temperature for consistency
                top_p=0.95,
            )
            
            planning_messages = [SystemMessage(content="You are a JSON generator. Return only valid JSON, nothing else.")]
            planning_messages.append(HumanMessage(content=retry_prompt))
            
            response = planning_llm.invoke(planning_messages)
            plan_text = response.content
            
            # Extract JSON
            start_idx = plan_text.find('{')
            end_idx = plan_text.rfind('}') + 1
            if start_idx >= 0 and end_idx > start_idx:
                json_content = plan_text[start_idx:end_idx]
                self.interview_plan = json.loads(json_content)
                print(f"Successfully created plan on retry with {len(self.interview_plan['questions'])} questions")
            else:
                # If still failing, generate questions individually
                self.generate_questions_individually()
        except Exception as e:
            print(f"Error in plan retry: {str(e)}")
            # Last resort: generate questions individually
            self.generate_questions_individually()
    
    def generate_questions_individually(self):
        """Generate interview questions individually as a last resort."""
        print("Generating questions individually...")
        
        questions = []
        
        # Create behavioral questions
        for i in range(self.behavioral_questions):
            try:
                # Generate a single behavioral question
                bq_prompt = f"Generate one challenging behavioral interview question for a {self.job_position} position. Return only the question text."
                bq_messages = [SystemMessage(content="You generate interview questions. Be concise and direct.")]
                bq_messages.append(HumanMessage(content=bq_prompt))
                
                bq_response = self.llm.invoke(bq_messages)
                question_text = bq_response.content.strip()
                
                questions.append({
                    "id": len(questions) + 1,
                    "type": "behavioral",
                    "question": question_text,
                    "follow_ups": []  # Will be generated dynamically
                })
            except Exception as e:
                print(f"Error generating behavioral question: {str(e)}")
                # Add a simple fallback question
                questions.append({
                    "id": len(questions) + 1, 
                    "type": "behavioral",
                    "question": f"Tell me about a challenging situation you faced in your career and how you handled it.",
                    "follow_ups": []
                })
        
        # Create technical questions for each skill
        for skill in self.skills[:self.technical_questions]:
            try:
                # Generate a technical question for this skill
                tq_prompt = f"Generate one challenging technical interview question about {skill} for a {self.job_position} position. Make it specific and test deep technical knowledge. Return only the question text."
                tq_messages = [SystemMessage(content="You generate technical interview questions. Be specific and challenging.")]
                tq_messages.append(HumanMessage(content=tq_prompt))
                
                tq_response = self.llm.invoke(tq_messages)
                question_text = tq_response.content.strip()
                
                questions.append({
                    "id": len(questions) + 1,
                    "type": "technical",
                    "skill": skill,
                    "question": question_text,
                    "follow_ups": []  # Will be generated dynamically
                })
            except Exception as e:
                print(f"Error generating technical question for {skill}: {str(e)}")
                # Add a simple fallback question
                questions.append({
                    "id": len(questions) + 1,
                    "type": "technical",
                    "skill": skill,
                    "question": f"How would you solve a complex problem related to {skill}?",
                    "follow_ups": []
                })
        
        # Add any custom questions
        for question in self.custom_questions:
            questions.append({
                "id": len(questions) + 1,
                "type": "custom",
                "question": question,
                "follow_ups": []
            })
        
        # Create the basic interview plan
        self.interview_plan = {
            "introduction": f"Hi, I am Mia, your interviewer. Welcome to your interview for the {self.job_position} position. Could you please tell me your name?",
            "questions": questions,
            "conclusion": "Thank you for your time today. The team will review your interview responses, and someone will be in touch about next steps. Best of luck with your job search!"
        }
    
    def validate_and_fix_plan(self):
        """Validate the interview plan and fix any issues."""
        if not self.interview_plan:
            self.generate_questions_individually()
            return
            
        # Check if the plan has the required keys
        required_keys = ["introduction", "questions", "conclusion"]
        for key in required_keys:
            if key not in self.interview_plan:
                print(f"Interview plan missing '{key}' - fixing")
                if key == "introduction":
                    self.interview_plan["introduction"] = f"Hi, I am Mia, your interviewer. Welcome to your interview for the {self.job_position} position. Could you please tell me your name?"
                elif key == "conclusion":
                    self.interview_plan["conclusion"] = "Thank you for your time today. The team will review your interview responses, and someone will be in touch about next steps. Best of luck with your job search!"
                elif key == "questions":
                    # If questions are missing, we need to generate them
                    self.generate_questions_individually()
                    return
        
        # Check if we have enough questions
        questions = self.interview_plan["questions"]
        if not isinstance(questions, list) or len(questions) < 1:
            print("Interview plan has invalid questions - generating new ones")
            self.generate_questions_individually()
            return
            
        # Ensure each question has the required fields
        for i, q in enumerate(questions):
            if not isinstance(q, dict) or "question" not in q:
                print(f"Question {i+1} is invalid - fixing")
                questions[i] = {
                    "id": i+1,
                    "type": questions[i].get("type", "general"),
                    "question": questions[i].get("question", "Tell me about your experience"),
                    "follow_ups": questions[i].get("follow_ups", [])
                }
                
            # Ensure follow_ups field exists (can be empty)
            if "follow_ups" not in q:
                q["follow_ups"] = []
                
            # Ensure technical questions have skill field
            if q.get("type") == "technical" and "skill" not in q:
                # Try to infer skill from question text
                for skill in self.skills:
                    if skill.lower() in q["question"].lower():
                        q["skill"] = skill
                        break
                else:
                    # No skill found, use a generic skill
                    q["skill"] = "technical"
        
        # Ensure questions have unique IDs
        for i, q in enumerate(questions):
            q["id"] = i+1
        
        # Count the number of behavioral and technical questions
        behavioral_count = sum(1 for q in questions if q.get("type") == "behavioral")
        technical_count = sum(1 for q in questions if q.get("type") == "technical")
        
        print(f"Plan validated: {behavioral_count} behavioral, {technical_count} technical questions")

    def create_follow_up_prompt(self) -> str:
        """Create a prompt for a follow-up question based on the candidate's response."""
        questions = self.interview_plan["questions"]
        current_question = questions[self.current_question_index]
        main_question = current_question["question"]
        question_type = current_question.get("type", "general")
        skill = current_question.get("skill", "") if question_type == "technical" else ""
        
        # Get the last candidate response
        last_response = ""
        if len(self.messages) >= 2 and isinstance(self.messages[-1], HumanMessage):
            last_response = self.messages[-1].content
        
        # Check if we have pre-planned follow-ups in the interview plan
        if "follow_ups" in current_question and len(current_question["follow_ups"]) > self.follow_up_index:
            # Use the pre-planned follow-up
            follow_up_text = current_question["follow_ups"][self.follow_up_index]
            print(f"Using planned follow-up: {follow_up_text[:30]}...")
            
            return f"""
            Ask this follow-up question naturally:
            "{follow_up_text}"
            
            Remember:
            - Do not provide feedback on their previous answer
            - Do not add explanations
            - Keep it conversational and professional
            """
        else:
            # Generate a dynamic follow-up based on question type
            print("Generating dynamic follow-up question...")
            
            if question_type == "technical":
                # Technical follow-up should challenge or probe deeper into technical knowledge
                return f"""
                The candidate has just answered this technical question about {skill}: 
                "{main_question}"
                
                Their response was: "{last_response[:200]}..."
                
                Based on their response, generate ONE specific technical follow-up question that:
                - Probes deeper into their technical knowledge of {skill}
                - Challenges any assumptions they made
                - Asks them to explain a specific part of their answer in more detail
                - OR asks how they would handle a related edge case or constraint
                
                The follow-up should be conversational but technically challenging.
                Do NOT provide feedback or evaluation of their answer.
                Return ONLY the follow-up question with no introduction or explanation.
                """
            else:
                # Behavioral follow-up should seek more specific details or reflection
                return f"""
                The candidate has just answered this question: "{main_question}"
                
                Their response was: "{last_response[:200]}..."
                
                Based on their response, generate ONE specific follow-up question that:
                - Asks for more specific details about the situation they described
                - Probes into their thought process or decision-making
                - Asks about what they learned or how they would approach it differently now
                
                The follow-up should be conversational and natural.
                Do NOT provide feedback or evaluation of their answer.
                Return ONLY the follow-up question with no introduction or explanation.
                """
    
    def should_ask_follow_up(self, response: str) -> bool:
        """Determine if we should ask a follow-up question based on the candidate's response."""
        # If response is too short, don't follow up
        if len(response.split()) < 20:
            return False
        
        # Get current question type
        current_q_type = ""
        if self.current_question_index < len(self.interview_plan["questions"]):
            current_q_type = self.interview_plan["questions"][self.current_question_index].get("type", "")
        
        # Technical questions get more follow-ups
        if current_q_type == "technical":
            # For technical questions, if we have pre-planned follow-ups, use them
            # Otherwise, do up to 2 dynamic follow-ups based on response quality
            if self.follow_up_index < 2:
                # Check response quality: looking for technical details, code samples, specifics
                has_details = any(kw in response.lower() for kw in 
                                ["code", "function", "algorithm", "complexity", "implement", 
                                 "design", "structure", "approach", "solution", "method"])
                
                # If the response has technical substance, follow up
                return has_details
        
        # For behavioral questions, only do one follow-up, if the response has enough substance
        if current_q_type == "behavioral" and self.follow_up_index < 1:
            # Look for signs of a substantive answer with details
            has_substance = any(kw in response.lower() for kw in 
                              ["experience", "project", "team", "challenge", "learned", 
                               "developed", "worked", "built", "created", "managed"])
            
            # If the response has substance, follow up
            return has_substance
        
        # For custom questions, use any pre-planned follow-ups, but limit dynamic ones
        if current_q_type == "custom" and self.follow_up_index < 1:
            return len(response.split()) > 30  # Only follow up if answer is substantial
        
        # Default: move on
        return False