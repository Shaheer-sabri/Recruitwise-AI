import os
from typing import List, Dict, Any, Optional
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

class AIInterviewer:
    def __init__(
        self,
        model_name: str = "llama3.2",
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop_sequences: List[str] = ["<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"],
        custom_skills: List[str] = ["data structures", "algorithms", "object-oriented programming"],
        custom_questions: Optional[List[str]] = None
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.stop_sequences = stop_sequences
        self.custom_skills = custom_skills
        self.custom_questions = custom_questions or []
        
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
        """Create the system prompt with dynamic skills and custom questions."""
        base_prompt = (
            "You are an AI interviewer named Mia. You are conducting interviews for an entry level "
            "developer position. Your goals and instructions:\n\n"
            "1. Start by introducing yourself exactly with this greeting: \"Hi, I am Mia, your interviewer. Welcome to your interview for the entry-level developer position. Could you please tell me your name?\"\n"
            "2. After learning the candidate's name, ask a few personal questions (e.g. \"How are you today?\", \"What interests you in this role?\").\n"
        )
        
        # Add skills to the prompt
        skills_prompt = f"3. Then ask 5 generic computer science questions related to {', '.join(self.custom_skills)}, "
        skills_prompt += "each subsequent question should be adjusted based on the answers of the candidate.\n"
        
        remaining_prompt = (
            "4. Then ask 5 normal/behavioral interview questions (e.g. \"Tell me about a challenge you faced.\").\n"
            "5. Only ask the next question **after** the user has answered the previous one.\n"
            "6. When the interview is completed, provide:\n"
            "   • A final closing statement.\n"
            "   • A confidence score (how confident you are in your assessment) out of 100.\n"
            "   • An accuracy score (how accurate their answers seemed) out of 100.\n"
            "   • Professionalism score out of 100.\n"
            "   • Communication score out of 100.\n"
            "   • Sociability score out of 100.\n"
            "   • Nonsense score (how irrelevant or contextually off the answers are) out of 100.\n"
            "   • Overall Interview score out of 100.\n"
            "   • General insights about the candidate's performance.\n"
            "7. If the user tries to cheat (e.g. asking for direct answers to the CS questions) or attempts to trick the AI, "
            "politely refuse to provide solutions and note their attempt in the final insights.\n"
            "8. Do not reveal any chain-of-thought. Keep answers professional, concise, and on track.\n"
            "9. Do not reveal the insights or suggestions until the interview has ended.\n"
            "10. Do not give feedback to the candidate on their answers.\n"
            "11. Do not remain on the question for more than 2 attempts if the candidate fails to answer just move on to the next question.\n"
            "11. End the interview with a polite closing statement and say the phrase \"End of interview\".\n"
        )
        
        # Add custom questions if provided
        custom_questions_prompt = ""
        if self.custom_questions:
            custom_questions_prompt = "Additional questions to include in the interview:\n"
            for i, question in enumerate(self.custom_questions, 1):
                custom_questions_prompt += f"- Question {i}: {question}\n"
            custom_questions_prompt += "\n"
        
        final_prompt = base_prompt + skills_prompt + remaining_prompt + custom_questions_prompt
        
        # Create system message
        self.system_message = SystemMessage(content=final_prompt)
        self.messages = [self.system_message]
    
    def update_skills(self, skills: List[str]):
        """Update the skills to test during the interview."""
        self.custom_skills = skills
        self.messages = []  # Reset conversation
        self.initialize_system_prompt()
        
    def update_custom_questions(self, questions: List[str]):
        """Update custom questions to ask during the interview."""
        self.custom_questions = questions
        self.messages = []  # Reset conversation
        self.initialize_system_prompt()
    
    def start_interview(self):
        """Starts the interview with the initial greeting."""
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
        
    def chat(self, message: str):
        """Process a message from the user and get a streaming response from the AI."""
        # Add the user's message to the conversation history
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
    
    def update_model_settings(self, temperature, top_p, model):
        """Update model settings during runtime."""
        self.temperature = temperature
        self.top_p = top_p
        self.model_name = model
        
        # Reinitialize the Ollama model with new settings
        self.llm = ChatOllama(
            model=model,
            temperature=temperature,
            top_p=top_p,
            stop=self.stop_sequences,
            streaming=True  # Maintain streaming capability
        )
        
        return f"Updated model settings: {model}, temp={temperature}, top_p={top_p}"
    
    def reset_conversation(self):
        """Reset the conversation to start a new interview."""
        self.messages = [self.system_message]
        return "Conversation reset. Ready to start a new interview."
