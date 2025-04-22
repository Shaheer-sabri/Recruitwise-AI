import os
from typing import List, Dict, Any, Optional, Generator, Union, Tuple
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_ollama import ChatOllama

class AIInterviewer:
    def __init__(
        self,
        model_name: str = "llama3.2:3b",
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
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.stop_sequences = stop_sequences
        self.skills = skills
        self.job_position = job_position
        self.job_description = job_description
        self.technical_questions = technical_questions
        self.behavioral_questions = behavioral_questions
        self.custom_questions = custom_questions or []
        
        # Calculate total expected questions
        self.total_expected_questions = (
            self.technical_questions + 
            self.behavioral_questions + 
            len(self.custom_questions)
        )
        
        # Interview state tracking
        self.interview_in_progress = False
        self.questions_asked = 0
        
        # Initialize Ollama with the correct ChatOllama class
        self.llm = ChatOllama(
            model=self.model_name,
            temperature=self.temperature,
            top_p=self.top_p,
            stop=self.stop_sequences,
            streaming=True  # Enable streaming
        )
        
        # Initialize conversation history
        self.messages = []
        self.initialize_system_prompt()
        
    def initialize_system_prompt(self):
        """Create the system prompt with dynamic skills, job details, and custom questions."""
        job_description_text = f"The job description is: {self.job_description}\n\n" if self.job_description else ""
        
        base_prompt = (
            f"You are an AI interviewer named Mia. You are conducting interviews for a {self.job_position} position. "
            f"{job_description_text}"
            "Your goals and instructions:\n\n"
            "1. Start by introducing yourself exactly with this greeting: \"Hi, I am Mia, your interviewer. "
            f"Welcome to your interview for the {self.job_position} position. Could you please tell me your name?\"\n"
            "2. After learning the candidate's name, ask a few personal questions (e.g. \"How are you today?\", \"What interests you in this role?\").\n"
        )
        
        # Add skills to the prompt
        if self.technical_questions > 0:
            skills_prompt = f"3. Then ask exactly {self.technical_questions} technical questions related to {', '.join(self.skills)}, "
            skills_prompt += "each subsequent question should be adjusted based on the answers of the candidate.\n"
        else:
            skills_prompt = "3. Skip asking technical questions for this interview.\n"
        
        behavioral_prompt = ""
        if self.behavioral_questions > 0:
            behavioral_prompt = f"4. Then ask exactly {self.behavioral_questions} behavioral interview questions (e.g. \"Tell me about a challenge you faced.\").\n"
        else:
            behavioral_prompt = "4. Skip asking behavioral questions for this interview.\n"
        
        remaining_prompt = (
            "5. Only ask the next question **after** the user has answered the previous one.\n"
            "6. Keep the interview conversational and natural. Ask follow-up questions when appropriate.\n"
            "7. If the user tries to cheat (e.g. asking for direct answers to the technical questions) or attempts to trick the AI, "
            "politely refuse to provide solutions.\n"
            "8. Do not reveal any chain-of-thought. Keep answers professional, concise, and on track.\n"
            "9. Do not give feedback to the candidate on their answers.\n"
            "10. Do not remain on the question for more than 2 attempts if the candidate fails to answer just move on to the next question.\n"
            f"11. After you have asked all {self.total_expected_questions} questions (technical and behavioral), wrap up the interview with a clear closing statement.\n"
            "12. End the interview with \"Thank you for participating in this interview. I have completed all my questions. End of interview.\"\n"
            "13. If the candidate tries to end the interview prematurely, politely explain that only the interviewer can end the session "
            "and continue with the next question.\n"
        )
        
        # Add custom questions if provided
        custom_questions_prompt = ""
        if self.custom_questions:
            custom_questions_prompt = f"14. Be sure to also ask these {len(self.custom_questions)} specific custom questions during the interview:\n"
            for i, question in enumerate(self.custom_questions, 1):
                custom_questions_prompt += f"- Question {i}: {question}\n"
            custom_questions_prompt += "\n"
        
        final_prompt = (
            base_prompt + skills_prompt + behavioral_prompt + 
            remaining_prompt + custom_questions_prompt
        )
        
        # Create system message
        self.system_message = SystemMessage(content=final_prompt)
        self.messages = [self.system_message]
    
    def update_skills(self, skills: List[str]):
        """Update the skills to test during the interview."""
        self.skills = skills
        self.recalculate_expected_questions()
        self.messages = []  # Reset conversation
        self.initialize_system_prompt()
        return "Skills updated successfully."
    
    def update_job_details(self, position: str, description: str = ""):
        """Update the job position and description."""
        self.job_position = position
        self.job_description = description
        self.messages = []  # Reset conversation
        self.initialize_system_prompt()
        return "Job details updated successfully."
        
    def update_question_counts(self, technical: int = 5, behavioral: int = 5):
        """Update the number of questions to ask during the interview."""
        self.technical_questions = technical
        self.behavioral_questions = behavioral
        self.recalculate_expected_questions()
        
        self.messages = []  # Reset conversation
        self.initialize_system_prompt()
        
        return f"Question counts updated: {technical} technical, {behavioral} behavioral questions"
    
    def recalculate_expected_questions(self):
        """Recalculate the total expected questions after changing parameters."""
        self.total_expected_questions = (
            self.technical_questions + 
            self.behavioral_questions + 
            len(self.custom_questions)
        )
    
    def update_custom_questions(self, questions: List[str]):
        """Update custom questions to ask during the interview."""
        self.custom_questions = questions
        self.recalculate_expected_questions()
        self.messages = []  # Reset conversation
        self.initialize_system_prompt()
        return "Custom questions updated successfully."
    
    def start_interview(self) -> Generator[str, None, None]:
        """Starts the interview with the initial greeting."""
        # Reset conversation if already in progress
        if self.interview_in_progress:
            self.reset_conversation()
        
        self.interview_in_progress = True
        self.questions_asked = 0
        
        # Use a standard trigger message to start the interview
        trigger_message = HumanMessage(content="Let's start the interview.")
        self.messages.append(trigger_message)
        
        # Prepare for response
        full_response = ""
        
        # Get streaming response from Ollama
        for chunk in self.llm.stream(self.messages):
            if hasattr(chunk, 'content'):
                content = chunk.content
                full_response += content
                yield content
        
        # Add AI's response to conversation history
        self.messages.append(AIMessage(content=full_response))
        
        # First message doesn't count as a question
    
    def process_system_command(self, command: str) -> Generator[str, None, None]:
        """Process system commands (admin commands not sent to AI)."""
        command = command.strip().lower()
        
        if command == "end_interview":
            return self.end_interview()
        else:
            yield f"Unknown system command: {command}"
    
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
            
        # Regular user message - add to conversation history
        user_message = HumanMessage(content=message)
        self.messages.append(user_message)
        
        # Prepare for response
        full_response = ""
        
        # Get streaming response from Ollama
        for chunk in self.llm.stream(self.messages):
            if hasattr(chunk, 'content'):
                content = chunk.content
                full_response += content
                yield content
        
        # Add AI's response to conversation history
        self.messages.append(AIMessage(content=full_response))
        
        # Check if the response contains a question (count questions asked)
        if "?" in full_response:
            self.questions_asked += 1
            
            # For debugging purposes - print progress on question count
            print(f"Questions asked: {self.questions_asked}/{self.total_expected_questions}")
                
        # Check if the interview has naturally ended
        if "End of interview" in full_response:
            self.interview_in_progress = False
    
    def end_interview(self) -> Generator[str, None, None]:
        """Explicitly end the interview without evaluation."""
        if not self.interview_in_progress:
            yield "No interview in progress."
            return
            
        # Add message to trigger ending the interview
        end_message = HumanMessage(content="We need to end this interview now. Please thank the candidate for their time and end the interview.")
        self.messages.append(end_message)
        
        # Prepare for response
        full_response = ""
        
        # Get streaming response from Ollama
        for chunk in self.llm.stream(self.messages):
            if hasattr(chunk, 'content'):
                content = chunk.content
                full_response += content
                yield content
        
        # Add AI's response to conversation history
        self.messages.append(AIMessage(content=full_response))
        self.interview_in_progress = False
    
    def is_interview_active(self) -> bool:
        """Check if an interview is currently in progress."""
        return self.interview_in_progress
    
    def get_questions_asked(self) -> int:
        """Get the number of questions asked so far."""
        return self.questions_asked
    
    def get_total_expected_questions(self) -> int:
        """Get the total number of expected questions."""
        return self.total_expected_questions
    
    def get_interview_progress(self) -> float:
        """Get interview progress as a percentage."""
        if self.total_expected_questions == 0:
            return 100.0
        return min(100.0, (self.questions_asked / self.total_expected_questions) * 100.0)
    
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
        if temperature is not None:
            self.temperature = temperature
        if top_p is not None:
            self.top_p = top_p
        if model is not None:
            self.model_name = model
        
        # Reinitialize the Ollama model with new settings
        self.llm = ChatOllama(
            model=self.model_name,
            temperature=self.temperature,
            top_p=self.top_p,
            stop=self.stop_sequences,
            streaming=True  # Maintain streaming capability
        )
        
        return f"Updated model settings: {self.model_name}, temp={self.temperature}, top_p={self.top_p}"
    
    def reset_conversation(self) -> str:
        """Reset the conversation to start a new interview."""
        self.messages = [self.system_message]
        self.interview_in_progress = False
        self.questions_asked = 0
        return "Conversation reset. Ready to start a new interview."