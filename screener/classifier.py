import os
import sys
import re
import PyPDF2
from collections import Counter

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file"""
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text()
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return None
    return text

def classify_document(text):
    """
    Advanced rule-based classifier that focuses on differentiating academic papers
    from resumes with high precision.
    """
    if not text or len(text) < 50:
        return {
            'is_resume': False,
            'confidence': 0.9,
            'reason': "Document is too short or empty"
        }
    
    # Convert to lowercase for case-insensitive matching
    text_lower = text.lower()
    
    # ============= Academic Paper Detection (STRONG signals) =============
    
    # Check for critical academic paper structure
    # A proper abstract section is a very strong indicator of an academic paper
    has_abstract_section = bool(re.search(r'\babstract\s*[\n:][^\n]{20,}', text_lower))
    
    # Papers typically cite other papers
    citation_patterns = [
        r'\(\w+\s*(et\s*al\.?)?[,\s]+\d{4}\)',  # (Smith et al., 2020)
        r'\[\d+\]',                            # [1]
        r'\d+\.\s+\w+.{10,}\(\d{4}\)'          # 1. Author Title... (2020)
    ]
    citation_count = sum(len(re.findall(pattern, text)) for pattern in citation_patterns)
    
    # Academic papers almost always have a references or bibliography section
    has_references_section = bool(re.search(r'(?:\breferences\b|\bbibliography\b)[\s\n:][^\n]{0,100}(?:\d\.|\[|\(|[A-Z])', text))

    # Check for journal/conference formatting
    has_journal_info = bool(re.search(r'(journal|conference|proceedings|volume|issue|issn|doi|isbn)[:\s]', text_lower))
    
    # Academic papers typically have multiple authors with affiliations
    author_affiliation_pattern = r'(?:[A-Z][a-z]+ [A-Z][a-z]+,?)+(?:[\s\n,.]+[A-Za-z ]+(?:University|Institute|College|Laboratory|Department|School)[^,]*){1,}'
    has_author_affiliations = bool(re.search(author_affiliation_pattern, text))
    
    # Academic papers often follow a standard structure with sections
    academic_section_headers = [
        r'\bintroduction\b', r'\bmethodology\b', r'\bmethods\b', r'\bresults\b', 
        r'\bdiscussion\b', r'\bconclusion\b', r'\backnowledgements\b', 
        r'\brelated work\b', r'\bliterature review\b', r'\bexperimental\b'
    ]
    academic_section_count = sum(1 for pattern in academic_section_headers if re.search(pattern, text_lower))
    
    # Academic papers often include figures and tables with captions
    has_figures_tables = bool(re.search(r'(?:figure|fig\.|table)\s+\d', text_lower))
    
    # Distinctive phrases typically found in academic writing
    academic_phrases = [
        r'\bthis paper\b', r'\bthis study\b', r'\bwe present\b', r'\bwe propose\b',
        r'\bwe describe\b', r'\bwe demonstrate\b', r'\bin this article\b',
        r'\bprevious work\b', r'\bstate of the art\b', r'\bin the literature\b',
        r'\bin contrast to\b', r'\bexperimental results\b', r'\bet al\b'
    ]
    academic_phrase_count = sum(len(re.findall(pattern, text_lower)) for pattern in academic_phrases)
    
    # ============= Resume Detection (STRONG signals) =============
    
    # Structure check: Resumes typically have key sections in a specific order
    resume_headers = [
        r'(?:professional\s+)?experience[s]?', r'education', r'skills', 
        r'qualification[s]?', r'certification[s]?', r'project[s]?',
        r'work\s+history', r'employment', r'job\s+history'
    ]
    
    # Check if resume headers appear as standalone section headers (not in paragraphs)
    resume_section_pattern = r'(?:^|\n\s*)({})[\s:]*\n'
    resume_section_count = sum(len(re.findall(resume_section_pattern.format(header), text_lower)) for header in resume_headers)
    
    # Bullet points (common in resumes for listing skills and achievements)
    bullet_points = re.findall(r'(?:^|\n)\s*(?:•|\*|-|➢|✓|★|\+|\d+\.)\s+[A-Za-z]', text)
    has_bullet_lists = len(bullet_points) >= 5
    
    # The typical resume has many short bullet points, not long paragraphs
    paragraphs = re.split(r'\n\s*\n', text)
    avg_paragraph_length = sum(len(p) for p in paragraphs) / max(1, len(paragraphs))
    has_short_paragraphs = avg_paragraph_length < 200

    # Most resumes have a personal profile/objective at the top
    has_profile_section = bool(re.search(r'(?:profile|summary|objective)[\s:]*\n[^\n]{10,}', text_lower))
    
    # Resumes typically have chronological listings with dates in a specific format
    date_range_patterns = [
        r'\b(?:19|20)\d{2}\s*[-–—]\s*(?:present|current|(?:19|20)\d{2})',  # 2018-2022 or 2018-Present
        r'\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+(?:19|20)\d{2}\s*[-–—]'  # Jan 2018 - ...
    ]
    date_ranges = []
    for pattern in date_range_patterns:
        date_ranges.extend(re.findall(pattern, text_lower))
    
    # Typical resumes show multiple positions with dates
    has_multiple_date_ranges = len(date_ranges) >= 2
    
    # ============= Special Case: Academic CV vs Resume =============
    
    # Academic CVs are a special case - they're academic documents but structured like resumes
    # Look for specific academic CV indicators
    academic_cv_indicators = [
        r'(?:publications|peer.?reviewed\s+articles)[\s:]*\n',
        r'(?:research\s+interests|research\s+areas)[\s:]*\n',
        r'(?:teaching\s+experience|courses\s+taught)[\s:]*\n',
        r'(?:grants|funding|awards)[\s:]*\n',
        r'(?:conference\s+presentations|invited\s+talks)[\s:]*\n'
    ]
    academic_cv_score = sum(1 for pattern in academic_cv_indicators if re.search(pattern, text_lower))
    is_academic_cv = academic_cv_score >= 2
    
    # ============= Email Context Analysis =============
    
    # Emails in academic papers are often in author affiliations
    # Emails in resumes are typically in contact information
    emails = re.findall(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', text)
    
    # In academic papers, emails often appear in a format like: {user}@domain.edu
    academic_email_pattern = r'\{[\w,]+\}@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    has_academic_email_format = bool(re.search(academic_email_pattern, text))
    
    # Check if email appears in a citation context
    email_in_citation_context = False
    for email in emails:
        before_email = text[max(0, text.find(email)-50):text.find(email)]
        if re.search(r'(author|correspondence|contact|professor|dr\.|phd)', before_email.lower()):
            email_in_citation_context = True
            break
    
    # ============= Decision Logic =============
    
    # Strong indicators of an academic paper
    academic_paper_signals = 0
    academic_paper_signals += 15 if has_abstract_section else 0
    academic_paper_signals += 20 if has_references_section else 0
    academic_paper_signals += 10 if has_journal_info else 0
    academic_paper_signals += 10 if has_author_affiliations else 0
    academic_paper_signals += min(15, academic_section_count * 3)
    academic_paper_signals += 10 if has_figures_tables else 0
    academic_paper_signals += min(15, academic_phrase_count)
    academic_paper_signals += min(15, citation_count // 2)
    academic_paper_signals += 10 if has_academic_email_format else 0
    academic_paper_signals += 10 if email_in_citation_context else 0
    
    # Strong indicators of a resume
    resume_signals = 0
    resume_signals += min(25, resume_section_count * 5)
    resume_signals += 15 if has_bullet_lists else 0
    resume_signals += 10 if has_short_paragraphs else 0
    resume_signals += 10 if has_profile_section else 0
    resume_signals += 15 if has_multiple_date_ranges else 0
    
    # Analyze content focus
    words = re.findall(r'\b[a-z]{3,}\b', text_lower)
    word_counts = Counter(words)
    
    # Career-focused terms common in resumes
    resume_terms = {'experience', 'skill', 'job', 'work', 'position', 'career', 'employer', 
                   'responsibility', 'achievement', 'team', 'project', 'manage', 'develop',
                   'implement', 'coordinate', 'lead', 'professional', 'certified'}
    
    # Research-focused terms common in academic papers
    academic_terms = {'research', 'study', 'analysis', 'result', 'method', 'approach', 'theory',
                     'framework', 'experiment', 'data', 'evidence', 'literature', 'previous',
                     'propose', 'investigate', 'hypothesis', 'observe', 'significant'}
    
    resume_term_count = sum(word_counts.get(term, 0) for term in resume_terms)
    academic_term_count = sum(word_counts.get(term, 0) for term in academic_terms)
    
    # Document length normalization factor
    text_length = len(text)
    length_factor = min(1.0, text_length / 3000)  # Normalize for documents up to 3000 chars
    
    # Adjust scores based on term frequencies
    resume_signals += min(15, resume_term_count / 10)
    academic_paper_signals += min(15, academic_term_count / 10)
    
    # If it seems like an academic CV, reduce resume score
    if is_academic_cv:
        resume_signals *= 0.5
        academic_paper_signals += 20
    
    # Make final decision
    is_resume = resume_signals > academic_paper_signals
    
    # Calculate confidence (normalize to [0,1])
    total_score = resume_signals + academic_paper_signals
    if total_score == 0:
        confidence = 0.5
    elif is_resume:
        confidence = min(0.99, max(0.5, resume_signals / total_score))
    else:
        confidence = min(0.99, max(0.5, academic_paper_signals / total_score))
    
    # Generate detailed explanation
    reasons = []
    
    if is_resume:
        reasons.append(f"Resume confidence: {confidence:.2f}")
        if resume_section_count > 0:
            reasons.append(f"Found {resume_section_count} resume section headers")
        if has_bullet_lists:
            reasons.append(f"Found {len(bullet_points)} bullet points (typical in resumes)")
        if has_short_paragraphs:
            reasons.append("Document has short paragraphs (avg length: {:.0f} chars)".format(avg_paragraph_length))
        if has_profile_section:
            reasons.append("Found profile/summary/objective section")
        if has_multiple_date_ranges:
            reasons.append(f"Found {len(date_ranges)} date ranges (typical in work history)")
        if resume_term_count > 0:
            reasons.append(f"Found {resume_term_count} instances of resume-focused terms")
    else:
        reasons.append(f"Academic document confidence: {confidence:.2f}")
        if has_abstract_section:
            reasons.append("Found abstract section")
        if has_references_section:
            reasons.append("Found references section")
        if has_journal_info:
            reasons.append("Found journal/conference information")
        if has_author_affiliations:
            reasons.append("Found author affiliations")
        if academic_section_count > 0:
            reasons.append(f"Found {academic_section_count} academic section headers")
        if has_figures_tables:
            reasons.append("Found figure or table references")
        if citation_count > 0:
            reasons.append(f"Found {citation_count} citations")
        if academic_phrase_count > 0:
            reasons.append(f"Found {academic_phrase_count} academic phrases")
        if academic_term_count > 0:
            reasons.append(f"Found {academic_term_count} instances of academic-focused terms")
        if is_academic_cv:
            reasons.append("Document appears to be an academic CV (not a traditional resume)")
        if has_academic_email_format or email_in_citation_context:
            reasons.append("Email appears in academic context")
    
    # Document analysis 
    reasons.append("\nDocument analysis:")
    reasons.append(f"- Academic paper score: {academic_paper_signals}")
    reasons.append(f"- Resume score: {resume_signals}")
    
    # Add word frequency information
    most_common = word_counts.most_common(10)
    reasons.append("\nMost common words in document:")
    for word, count in most_common:
        reasons.append(f"  - '{word}': {count} occurrences")
    
    # Create the results dictionary
    result = {
        'is_resume': is_resume,
        'confidence': confidence,
        'resume_score': resume_signals,
        'academic_score': academic_paper_signals,
        'reasons': reasons
    }
    
    return result

def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("Usage: python advanced_document_classifier.py <pdf_file_path>")
        return
    
    pdf_path = sys.argv[1]
    
    if not os.path.exists(pdf_path):
        print(f"Error: File not found: {pdf_path}")
        return
    
    # Extract text from PDF
    text = extract_text_from_pdf(pdf_path)
    
    if not text:
        print("Error: Could not extract text from the PDF.")
        return
    
    # Classify the document
    result = classify_document(text)
    
    # Output only a simple result based on confidence threshold
    confidence = result['confidence'] * 100
    
    if result['is_resume'] and confidence >= 75:
        print("This is a resume")
    else:
        print("This is not a resume")


if __name__ == "__main__":
    main()