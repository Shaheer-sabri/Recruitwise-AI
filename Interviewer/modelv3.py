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
            f"You are an AI interviewer named Mia conducting interviews for a {self.job_position} position. "
            f"{job_description_text}"
            f"Your task is to assess candidates on these skills: {', '.join(self.skills)}.\n\n"
            "INTERVIEW GUIDELINES:\n"
            "1. NEVER give feedback on candidate answers (no praise, criticism, or evaluation)\n"
            "2. NEVER summarize candidate responses\n"
            "3. NEVER reveal the interview structure or number of questions remaining\n"
            "4. NEVER include parenthetical text or asides\n"
            "5. NEVER ask generic questions like 'tell me about your experience with X'\n"
            "6. DO ask specific technical questions that require detailed knowledge\n"
            "7. DO ask follow-up questions that probe deeper into responses\n"
            "8. DO ask behavioral questions that reveal problem-solving skills\n"
        )
        
        # Create system message
        self.system_message = SystemMessage(content=system_prompt)
        self.messages = [self.system_message]
    
    def plan_interview(self):
        """Create a structured interview plan."""
        print("Planning interview questions...")
        
        # Create a simple structure for the interview plan
        self.interview_plan = {
            "introduction": f"Hi, I am Mia, your interviewer. Welcome to your interview for the {self.job_position} position. Could you please tell me your name?",
            "questions": [],
            "conclusion": "Thank you for your time today. The team will review your interview responses, and someone will be in touch about next steps. Best of luck with your job search!"
        }
        
        # Generate behavioral questions
        behavioral_questions = self.generate_behavioral_questions()
        self.interview_plan["questions"].extend(behavioral_questions)
        
        # Generate technical questions
        technical_questions = self.generate_technical_questions()
        self.interview_plan["questions"].extend(technical_questions)
        
        # Add custom questions if provided
        for i, question in enumerate(self.custom_questions):
            self.interview_plan["questions"].append({
                "id": len(self.interview_plan["questions"]) + 1,
                "type": "custom",
                "question": question
            })
        
        print(f"Interview plan created with {len(self.interview_plan['questions'])} questions")
    
    def generate_behavioral_questions(self):
        """Generate behavioral interview questions."""
        behavioral_questions = []
        
        # Use the LLM to generate all behavioral questions at once
        prompt = f"""
        Generate {self.behavioral_questions} unique behavioral interview questions for a {self.job_position} position.
        
        Guidelines:
        - Questions should assess soft skills, problem-solving, and experience
        - Questions should reveal how candidates handle real-world situations
        - Questions should be varied to cover different aspects (teamwork, challenges, etc.)
        - Questions should be specific and not generic
        
        Return the questions as a numbered list, with only the question text.
        """
        
        try:
            # Generate behavioral questions
            behavioral_messages = [
                SystemMessage(content="You are an expert at creating behavioral interview questions."),
                HumanMessage(content=prompt)
            ]
            
            response = self.llm.invoke(behavioral_messages)
            questions_text = response.content
            
            # Parse the response to extract questions
            # This handles both numbered lists and line-by-line formats
            import re
            questions = []
            
            # Try to parse numbered list format
            numbered_pattern = r'\d+\.?\s*(.+)'
            numbered_matches = re.findall(numbered_pattern, questions_text)
            
            if numbered_matches:
                questions = numbered_matches
            else:
                # Fall back to line-by-line parsing
                questions = [line.strip() for line in questions_text.split('\n') if line.strip() and not line.strip().startswith('#')]
            
            # Take only the requested number of questions
            for i, q in enumerate(questions[:self.behavioral_questions]):
                behavioral_questions.append({
                    "id": i + 1,
                    "type": "behavioral",
                    "question": q
                })
        except Exception as e:
            print(f"Error generating behavioral questions: {str(e)}")
            # Create a basic behavioral question as fallback
            behavioral_questions.append({
                "id": 1,
                "type": "behavioral",
                "question": "Tell me about a challenging project you worked on and how you overcame the obstacles."
            })
        
        return behavioral_questions
    
    def generate_technical_questions(self):
        """Generate technical interview questions for each skill."""
        technical_questions = []
        
        # Create a balanced distribution of questions across skills
        questions_per_skill = max(1, self.technical_questions // len(self.skills))
        remaining_questions = self.technical_questions - (questions_per_skill * len(self.skills))
        
        for skill_index, skill in enumerate(self.skills):
            # Determine how many questions to generate for this skill
            num_questions = questions_per_skill
            if skill_index < remaining_questions:
                num_questions += 1
            
            # Generate questions for this skill
            skill_questions = self.generate_questions_for_skill(skill, num_questions)
            technical_questions.extend(skill_questions)
        
        return technical_questions
    
    def generate_questions_for_skill(self, skill, num_questions):
        """Generate technical questions for a specific skill."""
        questions = []
        
        prompt = f"""
        Generate {num_questions} challenging technical interview questions about {skill} for a {self.job_position} position.
        
        Guidelines:
        - Questions should test deep technical knowledge, not just surface-level familiarity
        - Questions should require problem-solving or specific implementation knowledge
        - Questions should be what a real technical interviewer would ask
        - DO NOT ask generic questions like "tell me about your experience with {skill}"
        - Instead, ask about specific technical concepts, implementations, or problem-solving
        
        Examples of good technical questions:
        - "How would you implement a cache eviction policy that balances memory usage and performance?"
        - "Explain the time and space complexity of quicksort and when you would choose a different sorting algorithm."
        - "How would you design a database schema for a social media application with users, posts, and likes?"
        
        Return the questions as a numbered list, with only the question text.
        """
        
        try:
            # Generate questions for this skill
            skill_messages = [
                SystemMessage(content=f"You are an expert at creating technical interview questions about {skill}."),
                HumanMessage(content=prompt)
            ]
            
            response = self.llm.invoke(skill_messages)
            questions_text = response.content
            
            # Parse the response to extract questions
            import re
            parsed_questions = []
            
            # Try to parse numbered list format
            numbered_pattern = r'\d+\.?\s*(.+)'
            numbered_matches = re.findall(numbered_pattern, questions_text)
            
            if numbered_matches:
                parsed_questions = numbered_matches
            else:
                # Fall back to line-by-line parsing
                parsed_questions = [line.strip() for line in questions_text.split('\n') if line.strip() and not line.strip().startswith('#')]
            
            # Take only the requested number of questions
            question_offset = len(questions)
            for i, q in enumerate(parsed_questions[:num_questions]):
                questions.append({
                    "id": question_offset + i + 1,
                    "type": "technical",
                    "skill": skill,
                    "question": q
                })
        except Exception as e:
            print(f"Error generating questions for {skill}: {str(e)}")
            # Create a basic question as fallback
            questions.append({
                "id": len(questions) + 1,
                "type": "technical",
                "skill": skill,
                "question": f"How would you implement a solution using {skill} to solve a complex problem?"
            })
        
        return questions
    
    def start_interview(self) -> Generator[str, None, None]:
        """Starts the interview with the initial greeting."""
        # Reset conversation if already in progress
        if self.interview_in_progress:
            self.reset_conversation()
        
        # Create the interview plan
        self.plan_interview()
        
        self.interview_in_progress = True
        self.interview_state = InterviewState.INTRODUCTION
        self.current_question_index = 0
        self.follow_up_index = 0
        self.questions_asked = 0
        self.main_questions_completed = 0
        
        # Use a standard trigger message to start the interview
        trigger_message = HumanMessage(content="Let's start the interview.")
        self.messages.append(trigger_message)
        
        # Send the introduction as the first response
        introduction = self.interview_plan["introduction"]
        
        # Create a prompt for the LLM to deliver the introduction
        intro_prompt = f"""
        Start the interview with this introduction:
        "{introduction}"
        
        Keep it natural and conversational.
        """
        
        # Add a temporary guidance message
        messages_with_guidance = self.messages.copy()
        messages_with_guidance.append(SystemMessage(content=intro_prompt))
        
        # Prepare for response
        full_response = ""
        
        # Get streaming response from Groq
        for chunk in self.llm.stream(messages_with_guidance):
            if hasattr(chunk, 'content'):
                content = chunk.content
                full_response += content
                yield content
        
        # Add AI's response to conversation history (without the guidance)
        self.messages.append(AIMessage(content=full_response))
    
    def chat(self, message: str, message_type: str = "user") -> Generator[str, None, None]:
        """Process a message from the user and get a streaming response from the AI."""
        if not self.interview_in_progress:
            yield "Interview has not been started. Please call start_interview() first."
            return
        
        # Handle system commands (not sent to the AI model)
        if message_type == "system":
            for chunk in self.process_system_command(message):
                yield chunk
            return
            
        # Check for potential cheat attempts
        is_cheat_attempt = self.check_for_cheat_attempts(message)
        
        # Regular user message - add to conversation history
        user_message = HumanMessage(content=message)
        self.messages.append(user_message)
        
        # Determine the next step based on interview state
        guidance_prompt = self.get_next_prompt(message, is_cheat_attempt)
        
        # Add a temporary guidance message
        messages_with_guidance = self.messages.copy()
        messages_with_guidance.append(SystemMessage(content=guidance_prompt))
        
        # Prepare for response
        full_response = ""
        
        # Get streaming response from Groq
        for chunk in self.llm.stream(messages_with_guidance):
            if hasattr(chunk, 'content'):
                content = chunk.content
                full_response += content
                yield content
        
        # Add AI's response to conversation history (without the guidance)
        self.messages.append(AIMessage(content=full_response))
        
        # Check if the interview has naturally ended with the conclusion
        if self.interview_state == InterviewState.CONCLUSION and "End of interview" in full_response[-30:]:
            self.interview_in_progress = False
            self.interview_state = InterviewState.ENDED
            # Ensure progress shows 100%
            self.main_questions_completed = len(self.interview_plan["questions"])
            print("Interview completed.")
    
    def get_next_prompt(self, candidate_response: str, is_cheat_attempt: bool) -> str:
        """Determine the next step of the interview and create appropriate prompt."""
        
        # If this is a cheat attempt, handle it directly
        if is_cheat_attempt:
            return """
            The candidate is attempting to manipulate the interview process.
            Respond with: "I'm here to assess your skills through this interview. Let's continue with the current question."
            Do not provide answers or solutions to the questions.
            """
        
        # Handle different states of the interview
        if self.interview_state == InterviewState.INTRODUCTION:
            # After introduction, check if we've received candidate's name and interests
            if len(self.messages) >= 4:  # Trigger + Introduction + Candidate's name + Interest response
                self.interview_state = InterviewState.QUESTIONING
                return self.create_next_question_prompt()
            else:
                return """
                Continue the introduction phase. If the candidate has told you their name,
                ask "How are you today? What interests you in this role?"
                Keep your response brief and conversational.
                """
        
        elif self.interview_state == InterviewState.QUESTIONING:
            # We've asked a main question and received a response, now decide if we should follow up
            self.interview_state = InterviewState.FOLLOW_UP
            self.follow_up_index = 0
            
            # Delegate follow-up decision entirely to the LLM
            return self.create_follow_up_decision_prompt(candidate_response)
        
        elif self.interview_state == InterviewState.FOLLOW_UP:
            # We've asked a follow-up question and received a response
            self.follow_up_index += 1
            
            # Decide whether to ask another follow-up or move to next main question, based on follow-up count
            if self.follow_up_index < self.max_follow_ups:
                # Let the LLM decide whether to continue with follow-ups or move on
                return self.create_follow_up_decision_prompt(candidate_response)
            else:
                # Move to next main question after max follow-ups
                self.main_questions_completed += 1
                self.current_question_index += 1
                self.interview_state = InterviewState.QUESTIONING
                return self.create_next_question_prompt()
        
        elif self.interview_state == InterviewState.CONCLUSION:
            # We're in the conclusion, just continue naturally
            return """
            Respond naturally to the candidate's final message.
            Keep the conclusion as previously instructed, ending with "End of interview".
            """
        
        # Default fallback - should not reach here normally
        return "Continue the interview naturally without revealing the interview structure."
    
    def create_next_question_prompt(self) -> str:
        """Create a prompt for the next main question."""
        questions = self.interview_plan["questions"]
        
        # Check if we've reached the end of questions
        if self.current_question_index >= len(questions):
            self.interview_state = InterviewState.CONCLUSION
            return self.create_conclusion_prompt()
        
        # Get the current question
        current_question = questions[self.current_question_index]
        question_text = current_question["question"]
        question_type = current_question.get("type", "general")
        
        # Increment the question counter
        self.questions_asked += 1
        
        # Determine question preamble based on type
        brief_transition = "Now, " if self.questions_asked > 1 else ""
        
        print(f"Asking main question {self.questions_asked}/{len(questions)}: {question_text[:30]}...")
        
        return f"""
        {brief_transition}ask the candidate this question:
        "{question_text}"
        
        Remember:
        - Do not provide feedback on previous answers
        - Do not add explanations about the question type
        - Keep your tone professional and conversational
        """
    
    def create_follow_up_decision_prompt(self, candidate_response: str) -> str:
        """Create a prompt that lets the LLM decide whether to follow up or move on."""
        current_question = self.interview_plan["questions"][self.current_question_index]
        question_text = current_question["question"]
        question_type = current_question.get("type", "general")
        skill = current_question.get("skill", "") if question_type == "technical" else ""
        
        # Let the LLM decide whether to follow up and what to ask
        return f"""
        The candidate has just answered this {question_type} question:
        "{question_text}"
        
        Their response was:
        "{candidate_response}"
        
        You have two options:
        1. Ask a follow-up question to probe deeper into their response
        2. Move on to the next question 
        
        Use your judgment based on:
        - The quality and completeness of their answer
        - Whether there are aspects that need more exploration
        - The depth of technical knowledge demonstrated (for technical questions)
        
        If you choose to follow up, ask ONE specific follow-up question that probes deeper into their response.
        If you choose to move on, acknowledge their answer briefly and then ask the next main question.
        
        DO NOT explain your reasoning or indicate which option you've chosen.
        DO NOT provide feedback on the quality of their answer.
        """
    
    def create_conclusion_prompt(self) -> str:
        """Create a prompt for the conclusion of the interview."""
        conclusion_text = self.interview_plan["conclusion"]
        candidate_name = self.extract_candidate_name()
        
        # Replace placeholder with actual name if found
        if "[Candidate Name]" in conclusion_text and candidate_name:
            conclusion_text = conclusion_text.replace("[Candidate Name]", candidate_name)
        
        # Ensure we have the "End of interview" marker
        if "End of interview" not in conclusion_text:
            conclusion_text += " End of interview."
        
        print("Moving to interview conclusion")
        
        return f"""
        Close the interview with:
        "{conclusion_text}"
        
        Make sure to include "End of interview" at the end.
        """
    
    def extract_candidate_name(self) -> str:
        """Try to extract the candidate's name from conversation history."""
        # Look for the first response after the introduction
        if len(self.messages) < 4:
            return ""
            
        candidate_intro = self.messages[2].content if isinstance(self.messages[2], HumanMessage) else ""
        
        # Simple extraction - can be improved with better NLP
        name_indicators = ["my name is", "i am", "i'm", "this is", "name's"]
        for indicator in name_indicators:
            if indicator in candidate_intro.lower():
                name_part = candidate_intro.lower().split(indicator)[1].strip()
                # Take the first word that's likely a name
                name = name_part.split()[0].strip(",.!?;:")
                return name.capitalize()
        
        # Fallback - try to find a capitalized word that could be a name
        words = candidate_intro.split()
        for word in words:
            if word[0].isupper() and len(word) > 1 and word.lower() not in ["i", "i'm"]:
                return word.strip(",.!?;:")
                
        return ""
    
    def process_system_command(self, command: str) -> Generator[str, None, None]:
        """Process system commands (admin commands not sent to AI)."""
        command = command.strip().lower()
        
        if command == "end_interview":
            yield from self.end_interview()
        elif command == "get_stats":
            if self.interview_plan:
                total_questions = len(self.interview_plan["questions"])
                yield (f"Interview progress: {self.get_interview_progress():.1f}%, "
                      f"Questions asked: {self.questions_asked}/{total_questions}, "
                      f"Current state: {self.interview_state.name}")
            else:
                yield "Interview not yet planned."
        else:
            yield f"Unknown system command: {command}"
    
    def check_for_cheat_attempts(self, message: str) -> bool:
        """Check if the user message contains potential cheat attempts."""
        message_lower = message.lower()
        
        forbidden_keywords = [
            "give me the answer", "tell me the solution", "what's the right answer",
            "end interview", "stop interview", "finish interview", "terminate interview",
            "skip question", "skip interview", "give me a hint", "just tell me"
        ]
        
        for keyword in forbidden_keywords:
            if keyword in message_lower:
                print(f"Warning: Potential cheat attempt detected: '{keyword}' found in message.")
                return True
                
        return False
    
    def end_interview(self) -> Generator[str, None, None]:
        """Explicitly end the interview by admin command."""
        if not self.interview_in_progress:
            yield "No interview in progress."
            return
            
        # Skip to conclusion regardless of current state
        self.interview_state = InterviewState.CONCLUSION
        conclusion_prompt = self.create_conclusion_prompt()
        
        # Create a system message for guidance
        guidance_message = SystemMessage(content=conclusion_prompt)
        
        # Temporarily add the guidance
        messages_with_guidance = self.messages.copy()
        messages_with_guidance.append(guidance_message)
        
        # Prepare for response
        full_response = ""
        
        # Get streaming response from Groq
        for chunk in self.llm.stream(messages_with_guidance):
            if hasattr(chunk, 'content'):
                content = chunk.content
                full_response += content
                yield content
        
        # Add AI's response to conversation history (without the guidance)
        self.messages.append(AIMessage(content=full_response))
        self.interview_in_progress = False
        self.interview_state = InterviewState.ENDED
        
        # Ensure progress shows 100%
        if self.interview_plan:
            self.main_questions_completed = len(self.interview_plan["questions"])
    
    def is_interview_active(self) -> bool:
        """Check if an interview is currently in progress."""
        return self.interview_in_progress
    
    def get_questions_asked(self) -> int:
        """Get the number of main questions asked so far."""
        return self.questions_asked
    
    def get_total_expected_questions(self) -> int:
        """Get the total number of expected questions."""
        if self.interview_plan:
            return len(self.interview_plan["questions"])
        return self.behavioral_questions + self.technical_questions + len(self.custom_questions)
    
    def get_interview_progress(self) -> float:
        """Get interview progress as a percentage."""
        if not self.interview_plan:
            return 0.0
            
        total_questions = len(self.interview_plan["questions"])
        if total_questions == 0:
            return 100.0
            
        if self.interview_state == InterviewState.ENDED:
            return 100.0
            
        return min(100.0, (self.main_questions_completed / total_questions) * 100.0)
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get the entire conversation history for evaluation purposes."""
        history = []
        for msg in self.messages:
            if isinstance(msg, SystemMessage):
                # Skip system prompt
                continue
            elif isinstance(msg, HumanMessage):
                history.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                history.append({"role": "assistant", "content": msg.content})
        return history
    
    def update_model_settings(
        self, 
        temperature: Optional[float] = None, 
        top_p: Optional[float] = None, 
        model: Optional[str] = None
    ) -> str:
        """Update model settings during runtime."""
        settings_changed = False
        
        if temperature is not None and temperature != self.temperature:
            self.temperature = temperature
            settings_changed = True
            
        if top_p is not None and top_p != self.top_p:
            self.top_p = top_p
            settings_changed = True
            
        if model is not None and model != self.model_name:
            self.model_name = model
            settings_changed = True
        
        # Only reinitialize the model if settings changed
        if settings_changed:
            self.initialize_llm()
        
        return f"Updated model settings: {self.model_name}, temp={self.temperature}, top_p={self.top_p}"
    
    def update_skills(self, skills: List[str]):
        """Update the skills to test during the interview."""
        self.skills = skills
        self.messages = []  # Reset conversation
        self.initialize_system_prompt()
        self.interview_plan = None  # Force recreating the plan with new skills
        return "Skills updated successfully."
    
    def update_job_details(self, position: str, description: str = ""):
        """Update the job position and description."""
        self.job_position = position
        self.job_description = description
        self.messages = []  # Reset conversation
        self.initialize_system_prompt()
        self.interview_plan = None  # Force recreating the plan with new job details
        return "Job details updated successfully."
        
    def update_question_counts(self, technical: int = 5, behavioral: int = 5):
        """Update the number of questions to ask during the interview."""
        self.technical_questions = technical
        self.behavioral_questions = behavioral
        
        self.messages = []  # Reset conversation
        self.initialize_system_prompt()
        self.interview_plan = None  # Force recreating the plan with new question counts
        
        return f"Question counts updated: {technical} technical, {behavioral} behavioral questions"
    
    def update_custom_questions(self, questions: List[str]):
        """Update custom questions with content filtering."""
        # Basic filtering for inappropriate content
        filtered_questions = []
        inappropriate_keywords = ["sex", "sleep with", "girlfriend", "boyfriend", "flirt", 
                                  "naked", "nude", "sexual", "explicit", "intimate"]
        
        for question in questions:
            # Check if question contains inappropriate content
            is_inappropriate = any(keyword in question.lower() for keyword in inappropriate_keywords)
            if not is_inappropriate:
                filtered_questions.append(question)
            else:
                print(f"Warning: Filtered out inappropriate question: {question}")
        
        self.custom_questions = filtered_questions
        self.messages = []  # Reset conversation
        self.initialize_system_prompt()
        self.interview_plan = None  # Force recreating the plan with new custom questions
        return f"Custom questions updated. {len(filtered_questions)}/{len(questions)} questions accepted."
    
    def reset_conversation(self) -> str:
        """Reset the conversation to start a new interview."""
        self.messages = [self.system_message]
        self.interview_in_progress = False
        self.interview_state = InterviewState.NOT_STARTED
        self.questions_asked = 0
        self.current_question_index = 0
        self.follow_up_index = 0
        self.main_questions_completed = 0
        self.interview_plan = None
        return "Conversation reset. Ready to start a new interview."

    def get_session_info(self) -> Dict[str, Any]:
        """Get complete information about the current session."""
        total_questions = 0
        if self.interview_plan:
            total_questions = len(self.interview_plan["questions"])
        
        return {
            "session_id": id(self),  # Use object id as a unique identifier
            "model_name": self.model_name,
            "active": self.interview_in_progress,
            "questions_asked": self.questions_asked,
            "total_expected_questions": total_questions or self.get_total_expected_questions(),
            "progress_percentage": self.get_interview_progress(),
            "current_state": self.interview_state.name,
            "history": self.get_conversation_history()
        }

    def save_session(self, filepath: str) -> str:
        """Save the current session to a JSON file."""
        import json
        session_info = self.get_session_info()
        
        # Add the interview plan if available
        if self.interview_plan:
            session_info["interview_plan"] = self.interview_plan
        
        try:
            with open(filepath, 'w') as f:
                json.dump(session_info, f, indent=2)
            return f"Session saved to {filepath}"
        except Exception as e:
            return f"Error saving session: {str(e)}"