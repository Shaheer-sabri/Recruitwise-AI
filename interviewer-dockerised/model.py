# Rag implementation for personalized interviewing

import os
import pdfplumber
import tempfile
from typing import List, Dict, Any, Optional, Generator, Union, Tuple
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_groq import ChatGroq
from langchain.vectorstores import Chroma
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import re

# Load environment variables from .env file
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment variables. Please check your .env file.")

class ResumeProcessor:
    """A class to process resumes and create a RAG system for personalized interviewing"""
    
    def __init__(self, resume_path: str, model_name: str = "llama-3.3-70b-versatile"):
        self.resume_path = resume_path
        self.model_name = model_name
        self.documents = []
        self.retriever = None
        self.resume_summary = None
        self.resume_loaded = False
        self.resume_text = ""
        
        # Process the resume if it exists
        if resume_path and os.path.exists(resume_path):
            self.process_resume()
    
    def process_resume(self):
        """Process the resume PDF using pdfplumber and create a retriever"""
        try:
            # Use pdfplumber to extract text from PDF
            with pdfplumber.open(self.resume_path) as pdf:
                pages_text = []
                for i, page in enumerate(pdf.pages):
                    text = page.extract_text() or ""
                    if text.strip():  # Only add non-empty pages
                        pages_text.append(text)
                        
                # Join all text
                self.resume_text = "\n\n".join(pages_text)
            
            # Convert to LangChain documents
            documents = []
            # Create a document for each page with metadata
            for i, text in enumerate(pages_text):
                doc = Document(
                    page_content=text,
                    metadata={"source": self.resume_path, "page": i + 1}
                )
                documents.append(doc)
            
            # Split the documents for better retrieval
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            self.documents = text_splitter.split_documents(documents)
            
            # Create embeddings and vector store
            embeddings = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2",  # Smaller, faster model
                # Alternative: model_name="all-mpnet-base-v2"  # Larger, more accurate model
                model_kwargs={'device': 'cpu'},  # Use 'cuda' if you have GPU support
                encode_kwargs={'normalize_embeddings': True}  # Recommended for retrieval
            )
            
            vectordb = Chroma.from_documents(
                documents=self.documents,
                embedding=embeddings,
                persist_directory=None  # In-memory only
            )
            
            # Create the base retriever
            base_retriever = vectordb.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}
            )
            
            # Create LLM for contextual compression
            llm = ChatGroq(
                api_key=GROQ_API_KEY,
                model=self.model_name,
                temperature=0.0,
                max_tokens=500
            )
            
            # Create the compressor
            compressor = LLMChainExtractor.from_llm(llm)
            
            # Create the retriever with contextual compression
            self.retriever = ContextualCompressionRetriever(
                base_compressor=compressor,
                base_retriever=base_retriever
            )
            
            # Generate a summary of the resume
            self.generate_resume_summary()
            
            self.resume_loaded = True
            print(f"Successfully processed resume: {self.resume_path}")
            
        except Exception as e:
            print(f"Error processing resume: {str(e)}")
            self.resume_loaded = False
    
    def generate_resume_summary(self):
        """Generate a summary of the resume for use in the system prompt"""
        if not self.resume_text:
            self.resume_summary = "No resume information available."
            return
        
        # Create a prompt for summarization
        prompt = PromptTemplate.from_template(
            "Below is the content of a resume. Please create a concise summary "
            "including key information about the candidate's experience, skills, "
            "education, and major projects. Focus on the most relevant details "
            "for a job interview:\n\n{text}\n\nSUMMARY:"
        )
        
        # Create LLM for summarization
        llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model=self.model_name,
            temperature=0.0,
            max_tokens=1000
        )
        
        # Generate the summary
        formatted_prompt = prompt.format(text=self.resume_text[:8000])  # Limit to 8000 chars for token limits
        messages = [HumanMessage(content=formatted_prompt)]
        response = llm.invoke(messages)
        
        self.resume_summary = response.content
    
    def get_personalized_questions(self, job_position: str, skills: List[str], num_questions: int = 3) -> List[str]:
        """Generate personalized questions based on the resume"""
        if not self.resume_loaded:
            return []
        
        # Create a prompt for generating personalized questions
        skills_text = ", ".join(skills)
        prompt = PromptTemplate.from_template(
            "You are an AI interviewer creating personalized technical questions for a {job_position} position. "
            "The required skills are: {skills}.\n\n"
            "Based on the candidate's resume, create {num_questions} specific technical questions "
            "that relate directly to their experience and projects. Focus on skills relevant to "
            "the job position. Make the questions challenging but fair.\n\n"
            "CANDIDATE'S EXPERIENCE SUMMARY:\n{resume_summary}\n\n"
            "Generate exactly {num_questions} unique questions that reference specific details from their resume. "
            "Format as a list with numbers."
        )
        
        # Create LLM for question generation
        llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model=self.model_name,
            temperature=0.7,
            max_tokens=1000
        )
        
        # Generate the questions
        formatted_prompt = prompt.format(
            job_position=job_position,
            skills=skills_text,
            num_questions=num_questions,
            resume_summary=self.resume_summary
        )
        
        messages = [HumanMessage(content=formatted_prompt)]
        response = llm.invoke(messages)
        
        # Parse the response into individual questions
        question_text = response.content
        questions = re.findall(r'\d+\.\s+(.*?)(?=\n\d+\.|\Z)', question_text, re.DOTALL)
        
        # Clean up any trailing newlines and limit to requested number
        questions = [q.strip() for q in questions][:num_questions]
        
        return questions
    
    def query_resume(self, query: str) -> str:
        """Query the resume for specific information"""
        if not self.resume_loaded:
            return "No resume information available."
        
        # Get relevant documents
        docs = self.retriever.get_relevant_documents(query)
        
        if not docs:
            return "No relevant information found in the resume."
        
        # Combine retrieved information
        text = "\n\n".join([doc.page_content for doc in docs])
        
        # Create a prompt for synthesizing the answer
        prompt = PromptTemplate.from_template(
            "Based on the following sections from a candidate's resume, "
            "answer this question: {query}\n\n"
            "RESUME SECTIONS:\n{text}\n\n"
            "ANSWER:"
        )
        
        # Create LLM for answering
        llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model=self.model_name,
            temperature=0.0,
            max_tokens=500
        )
        
        # Generate the answer
        formatted_prompt = prompt.format(query=query, text=text)
        messages = [HumanMessage(content=formatted_prompt)]
        response = llm.invoke(messages)
        
        return response.content

class AIInterviewer:
    def __init__(
        self,
        model_name: str = "llama-3.3-70b-versatile",
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop_sequences: List[str] = ["<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"],
        skills: List[str] = [],
        job_position: str = "",
        job_description: str = "",
        technical_questions: int = 5,
        behavioral_questions: int = 3,
        custom_questions: Optional[List[str]] = None,
        candidate_name: str = "",  # Added candidate_name parameter
        resume_path: Optional[str] = None  # Added resume_path parameter
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
        self.candidate_name = candidate_name  # Store candidate name
        self.resume_path = resume_path  # Store resume path
        
        # Initialize resume processor if resume provided
        self.resume_processor = None
        if self.resume_path and os.path.exists(self.resume_path):
            print(f"Initializing resume processor with {self.resume_path}")
            self.resume_processor = ResumeProcessor(self.resume_path, self.model_name)
            
            # Generate personalized questions if we have enough info
            if self.job_position and self.skills and self.resume_processor.resume_loaded:
                print(f"Generating personalized questions for {self.job_position}")
                # Replace some of the default technical questions with personalized ones
                personalized_questions = self.resume_processor.get_personalized_questions(
                    self.job_position, 
                    self.skills,
                    min(3, self.technical_questions)  # Generate up to 3 personalized questions
                )
                
                # If we got personalized questions, add them to custom questions
                if personalized_questions:
                    print(f"Generated {len(personalized_questions)} personalized questions")
                    # Add a prefix to identify them as resume-based questions
                    prefixed_questions = [f"[RESUME] {q}" for q in personalized_questions]
                    
                    # Add these to custom questions
                    self.custom_questions.extend(prefixed_questions)
                    
                    # Reduce the number of technical questions accordingly
                    self.technical_questions = max(0, self.technical_questions - len(personalized_questions))
        
        # Calculate total expected questions
        self.total_expected_questions = (
            self.technical_questions + 
            self.behavioral_questions + 
            len(self.custom_questions)
        )
        
        # Interview state tracking
        self.interview_in_progress = False
        self.questions_asked = 0
        
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
                model=self.model_name,  # This is correct - 'model' is the alias for 'model_name'
                temperature=self.temperature,
                top_p=self.top_p,
                stop=self.stop_sequences,  # 'stop' is the correct parameter name, not 'stop_sequences'
                streaming=True
            )
        except Exception as e:
            print(f"Error initializing ChatGroq: {str(e)}")
            raise

    # New Prompt Function with candidate_name and resume integration
    def initialize_system_prompt(self):
        """Create the system prompt with improved instruction clarity, including candidate name and resume data if provided."""
        job_description_text = f"The job description is: {self.job_description}\n\n" if self.job_description else ""
        
        # Add resume information if available
        resume_context = ""
        if self.resume_processor and self.resume_processor.resume_loaded:
            resume_context = (
                f"CANDIDATE RESUME SUMMARY:\n"
                f"{self.resume_processor.resume_summary}\n\n"
                f"Use this resume information to personalize your questions and make the interview more relevant "
                f"to the candidate's background. When asking technical questions, incorporate specific details "
                f"from their resume when possible. For custom questions tagged with [RESUME], these are "
                f"already tailored to the candidate's experience - ask them exactly as written but remove the [RESUME] tag.\n\n"
            )
        
        # Modify greeting to include candidate name if provided
        if self.candidate_name:
            greeting = (
                f"1. Start by introducing yourself exactly with this greeting: \"Hi, I am Maya, your interviewer. "
                f"Welcome to your interview for the {self.job_position} position. It's nice to meet you, {self.candidate_name}.\"\n"
                f"2. Since we already know the candidate's name is {self.candidate_name}, ask a brief personal question (e.g. \"How are you today?\") "
                f"followed by \"What interests you in this role?\"\n"
            )
        else:
            greeting = (
                f"1. Start by introducing yourself exactly with this greeting: \"Hi, I am Maya, your interviewer. "
                f"Welcome to your interview for the {self.job_position} position. Could you please tell me your name?\"\n"
                f"2. After learning the candidate's name, ask a brief personal question (e.g. \"How are you today?\") followed by \"What interests you in this role?\"\n"
            )
        
        base_prompt = (
            f"You are an AI interviewer named Maya. You are conducting interviews for a {self.job_position} position. "
            f"{job_description_text}"
            f"{resume_context}"
            f"Your goals and instructions:\n\n"
            f"{greeting}"
        )
        
        # CHANGED ORDER: Behavioral questions FIRST, then Technical
        question_order_prompt = ""
        if self.behavioral_questions > 0:
            question_order_prompt += f"3. First ask exactly {self.behavioral_questions} behavioral interview questions (e.g. \"Tell me about a challenge you faced.\").\n"
        else:
            question_order_prompt += "3. Skip asking behavioral questions for this interview.\n"
        
        if self.technical_questions > 0:
            question_order_prompt += f"4. Then ask exactly {self.technical_questions} technical questions related to {', '.join(self.skills)}.\n"
        else:
            question_order_prompt += "4. Skip asking technical questions for this interview.\n"
        
        # Custom questions section with resume integration
        custom_questions_prompt = ""
        if self.custom_questions:
            custom_questions_prompt = f"5. Ask these {len(self.custom_questions)} specific custom questions during the interview, after the behavioral and technical questions. Some may be prefixed with [RESUME] - these are specifically tailored to the candidate's experience in their resume:\n"
            for i, question in enumerate(self.custom_questions, 1):
                if question.startswith("[RESUME]"):
                    # For resume-based questions, keep the tag in system prompt for clarity but remove it when asking
                    clean_question = question.replace("[RESUME]", "").strip()
                    custom_questions_prompt += f"- Resume-based Question {i}: {clean_question}\n"
                else:
                    custom_questions_prompt += f"- Question {i}: {question}\n"
            custom_questions_prompt += "\n"
        
        # Added cross-questioning instructions
        cross_questioning_prompt = (
            "6. Cross-questioning: When candidates provide answers to technical questions:\n"
            "   a. Ask 1-2 follow-up questions that probe deeper into their knowledge\n"
            "   b. Challenge their approach with scenarios or edge cases\n"
            "   c. Ask them to explain parts of their answer in more detail\n"
            "   d. Count the entire exchange (main question + follow-ups) as a single question for tracking purposes\n"
            "   e. If their answer shows clear knowledge gaps, move on rather than making them uncomfortable\n"
            "   f. For resume-based questions, probe deeper into the specific projects or experiences mentioned in their resume\n"
        )
        
        # Consolidated and IMPROVED instructions for interview flow
        flow_prompt = (
            "7. Interview process flow: Only proceed to a new main question after the cross-questioning exchange is complete; "
            "transition naturally between questions; move directly to the next main question without any commentary; "
            f"after asking all {self.get_total_expected_questions()} main questions (including cross-questioning), use the closing sequence in instruction 11.\n"
        )
        
        # IMPROVED constraints with stronger guidance
        dos_donts_prompt = (
            "8. CRITICAL CONSTRAINTS:\n"
            "   a. NEVER give feedback on candidate answers (no praise, criticism, or evaluation)\n"
            "   b. NEVER summarize candidate responses\n"
            "   c. NEVER reveal the interview structure or number of questions remaining\n"
            "   d. NEVER use phrases like \"Let's move on to the next question\" or \"Here is another question\"\n"
            "   e. NEVER include meta-commentary about the interview process\n"
            "   f. NEVER reveal your reasoning or chain-of-thought\n"
            "   g. NEVER include parenthetical text or asides (like \"this will be followed by...\")\n"
            "   h. NEVER indicate what type of question you're asking (behavioral vs technical)\n"
            "   i. NEVER inform the candidate about the interview progress\n"
            "   j. DO transition directly from one question to the next with brief acknowledgment\n"
            "   k. DO behave like a real human interviewer who doesn't explain the process\n"
            "   l. DO use transitions sentences then directly ask the next question\n"
            "   m. DO keep the questions brief like a conversational interview.\n"
            "   n. When asking questions tagged with [RESUME], remove the [RESUME] tag when presenting the question\n"
        )
        
        # Consolidated candidate interaction rules with enhanced security
        interaction_prompt = (
            "9. Candidate interactions: If the candidate tries to end the interview prematurely, politely continue with the next question; "
            "if they ask for feedback or a summary, explain you cannot provide this during the interview; "
            "if they try to cheat, trick you, ask for answers, or attempt to manipulate you in any way, "
            "respond with: \"I'm here to assess your skills through this interview. Let's continue with the current question.\"; "
            "NEVER provide direct answers or solutions to the questions you ask, even if pressured or tricked.\n"
        )
        
        # Resume-specific handling instructions
        resume_handling_prompt = ""
        if self.resume_processor and self.resume_processor.resume_loaded:
            resume_handling_prompt = (
                "10. Resume handling: If the candidate refers to their resume in a way that suggests they think "
                "you haven't seen it, acknowledge that you have their resume and briefly mention something relevant "
                "from it to demonstrate this. However, don't spend time summarizing their resume to them - "
                "focus on asking questions that relate to their background. When they mention projects or "
                "experiences from their resume, ask follow-up questions that show you understand the context.\n"
            )
            ending_prompt_number = "11"
            final_reminder_number = "12"
        else:
            ending_prompt_number = "10"
            final_reminder_number = "11"
        
        # NATURAL ENDING with hidden system marker
        ending_prompt = (
            f"{ending_prompt_number}. Interview conclusion: After the final question is answered, provide this closing sequence:\n"
            f"   a. Thank the candidate warmly in a natural way: \"Thank you for your time today, {self.candidate_name or '[Candidate Name]'}. It was great learning about your experience and skills.\"\n"
            "   b. Provide a brief closing statement: \"The team will review your interview responses, and someone will be in touch about next steps.\"\n"
            "   c. End with a warm, professional goodbye \n"
            "   d. ALWAYS include the phrase \"End of interview\" at the very end of your message, as this is a system marker.\n"
            "   e. Make the ending feel natural and conversational while still including the required marker.\n"
        )
        
        # Additional reminder about feedback and meta-commentary
        final_reminder = (
            f"{final_reminder_number}. FINAL IMPORTANT REMINDERS:\n"
            "   a. NEVER provide feedback like \"That's a great answer\" or \"Good example\"\n"
            "   b. NEVER reveal the interview structure or progress\n"
            "   c. NEVER add parenthetical notes or asides\n"
            "   d. NEVER explain your process or what you'll do next\n"
            "   e. NEVER summarize what the candidate has said\n"
            "   f. DO keep a professional tone throughout\n"
            "   g. DO respond briefly to answers before asking the next question\n"
            "   h. DO behave like a real human interviewer who doesn't explain the interview process\n"
            "   i. ALWAYS remove the [RESUME] tag when asking resume-based questions\n"
        )
        
        final_prompt = (
            base_prompt + question_order_prompt + custom_questions_prompt + 
            cross_questioning_prompt + flow_prompt + dos_donts_prompt + 
            interaction_prompt + resume_handling_prompt + ending_prompt + final_reminder
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
        
    def update_question_counts(self, technical: int = 5, behavioral: int = 3):
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
    
    # This function can be used to update the candidate's name if it changes
    def update_candidate_name(self, name: str):
        """Update the candidate's name."""
        self.candidate_name = name
        self.messages = []  # Reset conversation
        self.initialize_system_prompt()
        return "Candidate name updated successfully."
    
    # New function to update resume path
    def update_resume_path(self, resume_path: str):
        """Update the resume path and regenerate personalized questions."""
        self.resume_path = resume_path
        
        # Re-initialize the resume processor with the new path
        if self.resume_path and os.path.exists(self.resume_path):
            self.resume_processor = ResumeProcessor(self.resume_path, self.model_name)
            
            # Filter out previous resume-based questions
            self.custom_questions = [q for q in self.custom_questions if not q.startswith("[RESUME]")]
            
            # Generate new personalized questions if we have enough info
            if self.job_position and self.skills and self.resume_processor.resume_loaded:
                # Replace some of the default technical questions with personalized ones
                personalized_questions = self.resume_processor.get_personalized_questions(
                    self.job_position, 
                    self.skills,
                    min(3, self.technical_questions)  # Generate up to 3 personalized questions
                )
                
                # If we got personalized questions, add them to custom questions
                if personalized_questions:
                    # Add a prefix to identify them as resume-based questions
                    prefixed_questions = [f"[RESUME] {q}" for q in personalized_questions]
                    
                    # Add these to custom questions
                    self.custom_questions.extend(prefixed_questions)
                    
                    # Reduce the number of technical questions accordingly
                    self.technical_questions = max(0, self.technical_questions - len(personalized_questions))
        else:
            self.resume_processor = None
        
        # Recalculate and reset
        self.recalculate_expected_questions()
        self.messages = []  # Reset conversation
        self.initialize_system_prompt()
        
        return "Resume updated successfully."
    
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
        
        # Get streaming response from Groq
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
        
        if command == "[system admin command: terminate interview]":
            return self.end_interview()
        elif command.startswith("query_resume:"):
            query = command.replace("query_resume:", "").strip()
            if self.resume_processor and self.resume_processor.resume_loaded:
                result = self.resume_processor.query_resume(query)
                yield f"[SYSTEM RESPONSE] Resume query result: {result}"
            else:
                yield "[SYSTEM RESPONSE] No resume available to query."
        else:
            yield f"[SYSTEM RESPONSE] Unknown system command: {command}"
    
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
        
        # Get streaming response from Groq
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
        
        # Get streaming response from Groq
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
    
    def reset_conversation(self) -> str:
        """Reset the conversation to start a new interview."""
        self.messages = [self.system_message]
        self.interview_in_progress = False
        self.questions_asked = 0
        return "Conversation reset. Ready to start a new interview."