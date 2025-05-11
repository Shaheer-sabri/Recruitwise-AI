"""
Resume Matcher Logic Module
This module contains the core logic for analyzing resumes against job descriptions.
"""

import os
import tempfile
import numpy as np
import re
import pdfplumber
from sentence_transformers import SentenceTransformer
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Ensure NLTK resources are downloaded
def download_nltk_resources():
    """Download necessary NLTK resources."""
    nltk.download('punkt')
    nltk.download('stopwords')

class ResumeMatcher:
    """Class for matching resume to job descriptions."""
    
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """Initialize the resume matcher with model and resources."""
        # Download NLTK resources
        download_nltk_resources()
        
        # Load stop words
        self.stop_words = set(stopwords.words('english'))
        
        # Initialize the model
        try:
            self.model = SentenceTransformer(model_name)
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            # Fallback to a simpler model
            try:
                self.model = SentenceTransformer('paraphrase-MiniLM-L3-v2')
            except Exception as e2:
                print(f"Error loading fallback model: {e2}")
                self.model = None
        
        # Define tech skills
        self.tech_skills = set([
            "python", "java", "javascript", "typescript", "c++", "c#", "go", "ruby", "php",
            "html", "css", "react", "angular", "vue", "node.js", "express", "django", "flask",
            "spring", "hibernate", "aws", "azure", "gcp", "docker", "kubernetes", "terraform",
            "jenkins", "circleci", "git", "github", "gitlab", "bitbucket", "jira", "confluence",
            "sql", "mysql", "postgresql", "mongodb", "cassandra", "redis", "elasticsearch",
            "kafka", "rabbitmq", "rest", "graphql", "grpc", "oauth", "jwt", "tensorflow",
            "pytorch", "scikit-learn", "pandas", "numpy", "matplotlib", "seaborn", "tableau",
            "power bi", "excel", "word", "powerpoint", "agile", "scrum", "kanban", "waterfall",
            "github"
        ])
        
        # Define multi-word skill patterns
        self.skill_patterns = [
            r'\b(machine learning)\b', r'\b(deep learning)\b', r'\b(natural language processing)\b',
            r'\b(computer vision)\b', r'\b(data science)\b', r'\b(data analysis)\b',
            r'\b(front end)\b', r'\b(back end)\b', r'\b(full stack)\b', r'\b(devops)\b',
            r'\b(cloud computing)\b', r'\b(software development)\b', r'\b(web development)\b',
            r'\b(mobile development)\b', r'\b(database management)\b', r'\b(project management)\b'
        ]
    
    def extract_text_from_pdf(self, pdf_file):
        """Extract text from uploaded PDF file."""
        try:
            # Create a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                temp_file.write(pdf_file)
                temp_file_path = temp_file.name
            
            print(f"Created temporary file at: {temp_file_path}")
            
            # Open the file after the with block has closed the file handle
            text_blocks = []
            try:
                with pdfplumber.open(temp_file_path) as doc:
                    print(f"PDF opened successfully. Number of pages: {len(doc.pages)}")
                    for page in doc.pages:
                        try:
                            # Extract text from page
                            text = page.extract_text()
                            if text:
                                text_blocks.append(text)
                        except Exception as e:
                            print(f"Error processing page: {e}")
            except Exception as e:
                print(f"Error opening PDF with pdfplumber: {e}")
            finally:
                # Try to delete in a finally block to ensure cleanup
                try:
                    import time
                    time.sleep(0.5)  # Small delay to ensure file is not in use
                    os.unlink(temp_file_path)  # Delete the temp file
                    print(f"Temporary file deleted successfully: {temp_file_path}")
                except Exception as e:
                    print(f"Warning: Could not delete temporary file: {e}")
            
            result_text = "\n".join(text_blocks)
            print(f"Extracted {len(result_text)} characters of text from PDF")
            return result_text
        except Exception as e:
            print(f"Error extracting text from PDF: {e}")
            import traceback
            print(traceback.format_exc())
            return ""
    
    def preprocess_text(self, text):
        """Clean and preprocess text."""
        # Convert to lowercase
        text = text.lower()
        # Remove excess whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep spaces between words
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Simple word tokenization
        try:
            # Try NLTK tokenization
            tokens = word_tokenize(text)
            # Remove stopwords
            filtered_tokens = [word for word in tokens if word not in self.stop_words]
        except:
            # Fallback to simple whitespace tokenization
            tokens = text.split()
            # Basic English stopwords if NLTK fails
            basic_stopwords = {'a', 'an', 'the', 'and', 'or', 'but', 'if', 'because', 
                              'as', 'what', 'when', 'where', 'how', 'is', 'are', 'was', 
                              'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 
                              'does', 'did', 'to', 'at', 'in', 'on', 'by', 'for', 'with', 
                              'about', 'of', 'from'}
            filtered_tokens = [word for word in tokens if word not in basic_stopwords]
            
        return ' '.join(filtered_tokens)
    
    def extract_skills(self, text):
        """Extract technical skills from text."""
        words = set(re.findall(r'\b\w+\b', text.lower()))
        # Find skills that are in both the text and our skills list
        found_skills = words.intersection(self.tech_skills)
        
        # Look for bigrams and trigrams (multi-word skills)
        for pattern in self.skill_patterns:
            if re.search(pattern, text.lower()):
                found_skills.add(re.search(pattern, text.lower()).group())
                
        return list(found_skills)
    
    def calculate_semantic_similarity(self, resume_text, job_text):
        """Calculate semantic similarity between resume and job description."""
        if not resume_text or not job_text or self.model is None:
            return 0.0
        
        # Preprocess texts
        processed_resume = self.preprocess_text(resume_text)
        processed_job = self.preprocess_text(job_text)
        
        # Generate embeddings
        try:
            resume_embedding = self.model.encode([processed_resume])[0]
            job_embedding = self.model.encode([processed_job])[0]
            
            # Calculate cosine similarity
            dot_product = np.dot(resume_embedding, job_embedding)
            norm_resume = np.linalg.norm(resume_embedding)
            norm_job = np.linalg.norm(job_embedding)
            
            if norm_resume == 0 or norm_job == 0:
                return 0.0
                
            similarity = dot_product / (norm_resume * norm_job)
            return similarity
        except Exception as e:
            print(f"Error calculating similarity: {e}")
            return 0.0
    
    def calculate_skill_match(self, resume_skills, job_description):
        """Calculate skill match score."""
        if not resume_skills:
            return 0.0, [], [], []
            
        job_desc_lower = job_description.lower()
        
        # Extract skills from job description too
        jd_skills = self.extract_skills(job_description)
        
        # Direct matches (skills explicitly mentioned in JD)
        direct_matches = []
        for skill in resume_skills:
            if skill.lower() in job_desc_lower:
                direct_matches.append(skill)
        
        direct_match_score = len(direct_matches) / max(len(resume_skills), 1)
        
        # Skill overlap (comparing extracted skills from both)
        if jd_skills:
            overlap_skills = []
            for rs in resume_skills:
                for js in jd_skills:
                    if rs.lower() == js.lower():
                        overlap_skills.append(rs)
                        break
                        
            overlap_score = len(overlap_skills) / len(jd_skills)
        else:
            overlap_score = 0
            overlap_skills = []
            
        # Combined score with weightage
        final_score = (direct_match_score * 0.2) + (overlap_score * 0.8)
        
        return final_score, direct_matches, overlap_skills, jd_skills
    
    def analyze_resume(self, resume_file, job_description):
        """Full analysis of a resume against a job description."""
        # Extract text from resume
        resume_text = self.extract_text_from_pdf(resume_file)
        
        if not resume_text:
            return {
                "success": False,
                "error": "Could not extract text from the resume."
            }
        
        # Extract skills from resume
        resume_skills = self.extract_skills(resume_text)
        
        # Calculate semantic similarity
        semantic_score = self.calculate_semantic_similarity(resume_text, job_description)
        semantic_percent = round(semantic_score * 100, 2)
        
        # Calculate skill match
        skill_score, direct_matches, overlap_skills, jd_skills = self.calculate_skill_match(resume_skills, job_description)
        skill_percent = round(skill_score * 100, 2)
        
        # Calculate final score (80% semantic, 20% skills)
        final_score = (semantic_percent * 0.8) + (skill_percent * 0.2)
        final_score = round(final_score, 2)
        
        # Determine match level
        match_level = "Low Match"
        if final_score >= 80:
            match_level = "High Match"
        elif final_score >= 60:
            match_level = "Good Match"
        
        # Return results
        return {
            "success": True,
            "resume_text": resume_text,
            "resume_skills": resume_skills,
            "job_skills": jd_skills,
            "direct_matches": direct_matches,
            "overlap_skills": overlap_skills,
            "semantic_score": semantic_percent,
            "skill_score": skill_percent,
            "final_score": final_score,
            "match_level": match_level
        }
    

    def get_eligibility(self, resume_file, job_description):
        """Get eligibility score for a resume against a job description."""
        print(f"ResumeMatcher received PDF content, size: {len(resume_file)} bytes")
        
        result = self.analyze_resume(resume_file, job_description)
        
        if not result["success"]:
            return {
                "success": False,  # Python native boolean, not NumPy bool_
                "score": 0
            }
        
        score = result["final_score"]
        is_eligible = bool(score >= 35)  # Convert to Python native boolean
        
        return {
            "success": is_eligible,  # Now a Python native boolean
            "score": float(score)     # Ensure score is a native float
        }