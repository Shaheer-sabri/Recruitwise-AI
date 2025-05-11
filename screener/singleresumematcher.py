"""
Improved Resume Matcher Logic Module
This module contains the enhanced core logic for analyzing resumes against job descriptions.
"""

import os
import re
import numpy as np
from io import BytesIO
import pdfplumber
from sentence_transformers import SentenceTransformer
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from difflib import SequenceMatcher
import traceback

# Ensure NLTK resources are downloaded
def download_nltk_resources():
    """Download necessary NLTK resources."""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

class ResumeMatcher:
    """Enhanced class for matching resumes to job descriptions."""
    
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """Initialize the resume matcher with model and resources."""
        # Download NLTK resources
        download_nltk_resources()
        
        # Load stop words
        try:
            self.stop_words = set(stopwords.words('english'))
        except Exception as e:
            print(f"Error loading stopwords: {e}")
            # Basic English stopwords as fallback
            self.stop_words = {'a', 'an', 'the', 'and', 'or', 'but', 'if', 'because', 
                             'as', 'what', 'when', 'where', 'how', 'is', 'are', 'was', 
                             'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 
                             'does', 'did', 'to', 'at', 'in', 'on', 'by', 'for', 'with', 
                             'about', 'of', 'from'}
        
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
        
        # Initialize cache for performance
        self.cache = {}
        
        # Define the various skill sets
        self._initialize_skill_sets()
        
        # Define common section patterns
        self.section_patterns = {
            "education": [r'education', r'academic background', r'qualifications', r'academic', r'degree'],
            "experience": [r'experience', r'employment history', r'work history', r'professional background'],
            "skills": [r'skills', r'technical skills', r'competencies', r'expertise', r'proficiencies'],
            "projects": [r'projects', r'portfolio', r'achievements', r'accomplishments'],
            "summary": [r'summary', r'profile', r'objective', r'about', r'professional summary']
        }
    
    def _initialize_skill_sets(self):
        """Initialize various skill sets for different industries."""
        # Tech skills
        self.tech_skills = {
            "python", "java", "javascript", "typescript", "c++", "c#", "go", "ruby", "php",
            "html", "css", "react", "angular", "vue", "node.js", "express", "django", "flask",
            "spring", "hibernate", "aws", "azure", "gcp", "docker", "kubernetes", "terraform",
            "jenkins", "circleci", "git", "github", "gitlab", "bitbucket", "jira", "confluence",
            "sql", "mysql", "postgresql", "mongodb", "cassandra", "redis", "elasticsearch",
            "kafka", "rabbitmq", "rest", "graphql", "grpc", "oauth", "jwt", "tensorflow",
            "pytorch", "scikit-learn", "pandas", "numpy", "matplotlib", "seaborn", "tableau",
            "power bi", "excel", "word", "powerpoint", "agile", "scrum", "kanban", "waterfall"
        }
        
        # Hospitality skills
        self.hospitality_skills = {
            "customer service", "hospitality", "front desk", "reception", "check-in", "check-out",
            "guest relations", "housekeeping", "hotel management", "tourism", "booking", 
            "reservation", "concierge", "guest experience", "food service", "event planning",
            "catering", "hotel operations", "guest satisfaction", "property management system",
            "opera", "travel", "accommodation", "lodging", "guest services"
        }
        
        # Business/management skills
        self.business_skills = {
            "leadership", "management", "strategic planning", "project management", "team building",
            "budgeting", "forecasting", "analytics", "business development", "negotiation",
            "marketing", "sales", "client relations", "stakeholder management", "operations",
            "process improvement", "problem solving", "decision making", "time management",
            "resource allocation", "risk management", "quality assurance", "communication"
        }
        
        # Combined industry skills for detection
        self.industry_skills = {
            "technology": self.tech_skills,
            "hospitality": self.hospitality_skills,
            "business": self.business_skills
        }
        
        # Multi-word skill patterns that won't be found through simple word matching
        self.skill_patterns = [
            r'\b(machine learning)\b', r'\b(deep learning)\b', r'\b(natural language processing)\b',
            r'\b(computer vision)\b', r'\b(data science)\b', r'\b(data analysis)\b',
            r'\b(front end)\b', r'\b(back end)\b', r'\b(full stack)\b', r'\b(devops)\b',
            r'\b(cloud computing)\b', r'\b(software development)\b', r'\b(web development)\b',
            r'\b(mobile development)\b', r'\b(database management)\b', r'\b(project management)\b',
            r'\b(customer experience)\b', r'\b(user experience)\b', r'\b(digital marketing)\b'
        ]
        
        # Context patterns for skill extraction
        self.context_patterns = [
            r'(?:proficient|expertise|skilled|experience|knowledge)\s+(?:in|with|of)\s+([\w\s]+)',
            r'(?:familiar|worked|trained)\s+(?:in|with)\s+([\w\s]+)',
            r'(?:specializ(?:ed|ing)|certified)\s+(?:in)\s+([\w\s]+)'
        ]
    
    def ensure_serializable(self, obj):
        """Recursively ensure all values in a dictionary or list are JSON serializable."""
        if isinstance(obj, dict):
            return {k: self.ensure_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.ensure_serializable(item) for item in obj]
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return self.ensure_serializable(obj.tolist())
        else:
            return obj
    
    def extract_text_from_pdf(self, pdf_file):
        """Extract text from uploaded PDF file using BytesIO for better reliability."""
        try:
            # Use BytesIO to avoid file system operations
            pdf_bytes = BytesIO(pdf_file)
            
            text_blocks = []
            with pdfplumber.open(pdf_bytes) as doc:
                print(f"PDF opened successfully. Number of pages: {len(doc.pages)}")
                for page in doc.pages:
                    try:
                        text = page.extract_text() or ""
                        if text:
                            text_blocks.append(text)
                    except Exception as e:
                        print(f"Error processing page: {e}")
            
            result_text = "\n".join(text_blocks)
            print(f"Extracted {len(result_text)} characters of text from PDF")
            return result_text
        except Exception as e:
            print(f"Error extracting text from PDF: {e}")
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
            filtered_tokens = [word for word in tokens if word not in self.stop_words]
            
        return ' '.join(filtered_tokens)
    
    def extract_sections(self, text):
        """Extract sections from a document."""
        sections = {}
        current_section = "general"
        sections[current_section] = []
        
        lines = text.split('\n')
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
                
            # Check if this line starts a new section
            section_match = False
            for section, patterns in self.section_patterns.items():
                if any(re.search(pattern, line.lower()) for pattern in patterns):
                    current_section = section
                    sections[current_section] = []
                    section_match = True
                    break
            
            # Add line to current section
            if not section_match:  # Skip section headers
                sections[current_section].append(line)
        
        # Convert section content from list to string
        for section in sections:
            sections[section] = "\n".join(sections[section])
        
        return sections
    
    def detect_industry(self, text):
        """Detect the likely industry based on skills mentioned in the text."""
        text_lower = text.lower()
        industry_scores = {}
        
        # Count skills from each industry
        for industry, skills in self.industry_skills.items():
            count = sum(1 for skill in skills if skill in text_lower)
            industry_scores[industry] = count
        
        # Get the industry with the highest score
        if not industry_scores:
            return "general"
            
        max_industry = max(industry_scores.items(), key=lambda x: x[1])
        
        # If no significant match, return general
        if max_industry[1] < 3:
            return "general"
            
        return max_industry[0]
    
    def extract_skills(self, text, industry=None):
        """Extract technical skills from text using simple word matching."""
        if not text:
            return []
            
        # Get words from text
        words = set(re.findall(r'\b\w+\b', text.lower()))
        
        # Initialize found skills
        found_skills = set()
        
        # If industry is provided, use industry-specific skills
        if industry and industry in self.industry_skills:
            found_skills.update(words.intersection(self.industry_skills[industry]))
        else:
            # Try all industry skills
            for industry_skills in self.industry_skills.values():
                found_skills.update(words.intersection(industry_skills))
        
        # Look for bigrams and trigrams (multi-word skills)
        for pattern in self.skill_patterns:
            matches = re.finditer(pattern, text.lower())
            for match in matches:
                found_skills.add(match.group())
        
        return list(found_skills)
    
    def extract_skills_with_context(self, text, industry=None):
        """Enhanced skill extraction with context awareness."""
        if not text:
            return []
            
        # Extract skills using basic method first
        skills = set(self.extract_skills(text, industry))
        
        # Extract skills using context patterns
        for pattern in self.context_patterns:
            matches = re.finditer(pattern, text.lower())
            for match in matches:
                if len(match.groups()) > 0:
                    skill_phrase = match.group(1).strip()
                    # Clean up the skill phrase
                    skill_phrase = re.sub(r'\b(and|with|using|in|of|the|for|to|a|an)\s*$', '', skill_phrase).strip()
                    if skill_phrase and len(skill_phrase) > 2 and not any(w in self.stop_words for w in skill_phrase.split()):
                        skills.add(skill_phrase)
        
        return list(skills)
    
    def _base_similarity(self, text1, text2):
        """Base semantic similarity calculation."""
        if not text1 or not text2 or self.model is None:
            return 0.0
        
        # Preprocess texts
        processed_text1 = self.preprocess_text(text1)
        processed_text2 = self.preprocess_text(text2)
        
        # Generate embeddings
        try:
            text1_embedding = self.model.encode([processed_text1])[0]
            text2_embedding = self.model.encode([processed_text2])[0]
            
            # Calculate cosine similarity
            dot_product = np.dot(text1_embedding, text2_embedding)
            norm_text1 = np.linalg.norm(text1_embedding)
            norm_text2 = np.linalg.norm(text2_embedding)
            
            if norm_text1 == 0 or norm_text2 == 0:
                return 0.0
                
            similarity = dot_product / (norm_text1 * norm_text2)
            # Convert to native Python float
            return float(similarity)
        except Exception as e:
            print(f"Error calculating similarity: {e}")
            return 0.0
    
    def calculate_semantic_similarity(self, resume_text, job_text):
        """Enhanced semantic similarity with section weighting."""
        # Use cache if available
        cache_key = f"sem_{hash(resume_text)}_{hash(job_text)}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Calculate overall similarity
        full_sim = self._base_similarity(resume_text, job_text)
        
        # Extract sections
        resume_sections = self.extract_sections(resume_text)
        job_sections = self.extract_sections(job_text)
        
        # Initialize weights and scores
        section_scores = [(full_sim, 0.6)]  # Overall similarity with 60% weight
        
        # Compare job requirements with resume skills if available
        if 'skills' in resume_sections and 'skills' in job_sections:
            skills_sim = self._base_similarity(resume_sections['skills'], job_sections['skills'])
            section_scores.append((skills_sim, 0.2))  # 20% weight
        
        # Compare job experience with resume experience if available
        if 'experience' in resume_sections and 'experience' in job_sections:
            exp_sim = self._base_similarity(resume_sections['experience'], job_sections['experience'])
            section_scores.append((exp_sim, 0.2))  # 20% weight
        
        # Calculate weighted average
        weighted_sum = sum(score * weight for score, weight in section_scores)
        total_weight = sum(weight for _, weight in section_scores)
        
        similarity = weighted_sum / total_weight
        
        # Convert to Python float to ensure JSON serialization
        similarity = float(similarity)
        
        # Cache the result
        self.cache[cache_key] = similarity
        return similarity
    
    def calculate_skill_match(self, resume_skills, job_description):
        """Calculate skill match with fuzzy matching and improved weighting."""
        if not resume_skills:
            return 0.0, [], [], []
        
        # Use cache if available
        cache_key = f"skill_{hash(str(resume_skills))}_{hash(job_description)}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Extract industry from job description
        industry = self.detect_industry(job_description)
        
        # Extract skills from job description
        jd_skills = self.extract_skills_with_context(job_description, industry)
        
        # If no skills found in job description, return zero match
        if not jd_skills:
            self.cache[cache_key] = (0.0, [], [], [])
            return 0.0, [], [], []
        
        # Direct matches (skills explicitly mentioned in JD)
        direct_matches = []
        for skill in resume_skills:
            if skill.lower() in job_description.lower():
                direct_matches.append(skill)
        
        # Fuzzy matching for skills
        def similarity(a, b):
            return SequenceMatcher(None, a.lower(), b.lower()).ratio()
        
        fuzzy_matches = []
        for rs in resume_skills:
            if rs in direct_matches:
                continue  # Skip already matched skills
                
            for js in jd_skills:
                # Check for fuzzy match with 80% similarity threshold
                if similarity(rs, js) > 0.8:
                    fuzzy_matches.append(rs)
                    break
        
        # All matched skills (direct + fuzzy)
        all_matches = direct_matches + fuzzy_matches
        
        # Calculate scores
        if len(resume_skills) == 0:
            direct_match_score = 0.0
        else:
            direct_match_score = len(direct_matches) / len(resume_skills)
        
        if len(jd_skills) == 0:
            coverage_score = 0.0
        else:
            coverage_score = len(all_matches) / len(jd_skills)
        
        # Combined score with improved weighting
        # Coverage is more important than direct matches
        final_score = (direct_match_score * 0.3) + (coverage_score * 0.7)
        
        # Convert to Python float
        final_score = float(final_score)
        
        # Cache and return the result
        result = (final_score, direct_matches, fuzzy_matches, jd_skills)
        self.cache[cache_key] = result
        return result
    
    def analyze_resume(self, resume_file, job_description):
        """Enhanced resume analysis with industry detection."""
        try:
            # Extract text from resume
            resume_text = self.extract_text_from_pdf(resume_file)
            
            if not resume_text:
                return {
                    "success": False,
                    "error": "Could not extract text from the resume. The PDF may be encrypted, scanned, or in an unsupported format."
                }
            
            # Detect industry
            industry = self.detect_industry(job_description)
            print(f"Detected industry: {industry}")
            
            # Extract skills from resume with industry context
            resume_skills = self.extract_skills_with_context(resume_text, industry)
            
            # Calculate semantic similarity
            semantic_score = self.calculate_semantic_similarity(resume_text, job_description)
            semantic_percent = round(float(semantic_score) * 100, 2)
            
            # Calculate skill match
            skill_score, direct_matches, fuzzy_matches, jd_skills = self.calculate_skill_match(resume_skills, job_description)
            skill_percent = round(float(skill_score) * 100, 2)
            
            # Extract sections
            resume_sections = self.extract_sections(resume_text)
            
            # Calculate final score (weighted by industry)
            if industry == "technology":
                # Tech jobs: 70% semantic, 30% skills
                semantic_weight = 0.7
                skill_weight = 0.3
            elif industry == "hospitality":
                # Hospitality: 80% semantic, 20% skills
                semantic_weight = 0.8
                skill_weight = 0.2
            else:
                # Default: 75% semantic, 25% skills
                semantic_weight = 0.75
                skill_weight = 0.25
                
            final_score = (semantic_percent * semantic_weight) + (skill_percent * skill_weight)
            final_score = round(float(final_score), 2)
            
            # Determine match level with variable thresholds by industry
            if industry == "technology":
                # Tech jobs have higher standards
                if final_score >= 80:
                    match_level = "High Match"
                elif final_score >= 60:
                    match_level = "Good Match"
                else:
                    match_level = "Low Match"
            else:
                # Standard thresholds
                if final_score >= 80:
                    match_level = "High Match"
                elif final_score >= 60:
                    match_level = "Good Match"
                else:
                    match_level = "Low Match"
            
            # Extract education details if available
            education = resume_sections.get('education', '')
            
            # Create result dictionary
            result = {
                "success": True,
                "resume_text": resume_text,
                "industry": industry,
                "resume_skills": resume_skills,
                "job_skills": jd_skills,
                "direct_matches": direct_matches,
                "fuzzy_matches": fuzzy_matches,
                "education": education,
                "semantic_score": semantic_percent,
                "skill_score": skill_percent,
                "weights": {
                    "semantic": float(semantic_weight),
                    "skill": float(skill_weight)
                },
                "final_score": final_score,
                "match_level": match_level
            }
            
            # Ensure all values are JSON serializable
            result = self.ensure_serializable(result)
            
            return result
            
        except Exception as e:
            print(f"Error in resume analysis: {e}")
            print(traceback.format_exc())
            return {
                "success": False,
                "error": f"Analysis error: {str(e)}"
            }
    
    def get_eligibility(self, resume_file, job_description):
        """Enhanced eligibility check with better error handling."""
        try:
            print(f"ResumeMatcher received PDF content, size: {len(resume_file)} bytes")
            
            result = self.analyze_resume(resume_file, job_description)
            
            if not result["success"]:
                return {
                    "success": False,
                    "score": 0,
                    "error": result.get("error", "Unknown error during analysis")
                }
            
            score = result["final_score"]
            industry = result.get("industry", "general")
            
            # Variable eligibility threshold by industry
            if industry == "technology":
                threshold = 40  # Higher for tech jobs
            elif industry == "hospitality":
                threshold = 30  # Lower for hospitality
            else:
                threshold = 35  # Default
                
            is_eligible = bool(score >= threshold)
            
            # Enhanced response with more details
            response = {
                "success": is_eligible,
                "score": float(score),  # Ensure native Python float
                "industry": industry,
                "threshold": threshold,
                "matches": {
                    "skills": result.get("direct_matches", []) + result.get("fuzzy_matches", []),
                    "skill_score": float(result.get("skill_score", 0)),  # Ensure native Python float
                    "semantic_score": float(result.get("semantic_score", 0))  # Ensure native Python float
                }
            }
            
            # Ensure all values are JSON serializable
            response = self.ensure_serializable(response)
            
            return response
            
        except Exception as e:
            print(f"Error in eligibility check: {e}")
            print(traceback.format_exc())
            return {
                "success": False,
                "score": 0,
                "error": f"Eligibility error: {str(e)}"
            }
    
    def rank_resumes(self, resume_files, job_description):
        """Rank multiple resumes against a job description."""
        results = []
        
        for resume_file in resume_files:
            result = self.analyze_resume(resume_file, job_description)
            if result["success"]:
                results.append(result)
        
        # Sort by final score (descending)
        ranked_results = sorted(results, key=lambda x: x["final_score"], reverse=True)
        
        # Ensure results are serializable
        ranked_results = self.ensure_serializable(ranked_results)
        
        return ranked_results