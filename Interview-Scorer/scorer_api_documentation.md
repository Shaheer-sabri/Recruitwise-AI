# Interview Scorer API Documentation

## Overview
The Interview Scorer API evaluates interview transcripts against job descriptions using language models.
It provides scores for specific skills and an overall evaluation of the candidate.

**Base URL:** `http://localhost:8000`

## Endpoints

### POST /evaluate
Evaluates an interview transcript against a job description for specific skills.

#### Request Body

| Parameter | Type | Description | Required |
|-----------|------|-------------|----------|
| job_description | string | The full job description text | Yes |
| interview_transcript | object | The interview transcript with history of exchanges | Yes |
| skills | array | List of skills to evaluate the candidate on | Yes |

#### Example Request
```json
{
  "job_description": "We are looking for a skilled Backend Software Engineer...",
  "interview_transcript": {
    "history": [
      {
        "role": "assistant", 
        "content": "Hello! I'll be interviewing you today for the Backend Software Engineer position."
      },
      {
        "role": "user",
        "content": "I've been working with Python for about 4 years now."
      }
    ]
  },
  "skills": ["Python", "REST APIs", "System Design"]
}
```

### Example Output
```json
{
  "overall_score": 8,
  "skill_scores": {
    "Python": 9,
    "REST APIs": 7,
    "System Design": 8
  },
  "skill_justifications": {
    "Python": "The candidate demonstrated strong knowledge of Python with clear examples of implementing microservices using Flask and FastAPI. They showed advanced understanding by explaining how they optimized a data processing pipeline using asyncio for parallel processing.",
    "REST APIs": "The candidate showed competent knowledge of REST API design principles, including the use of appropriate HTTP methods and stateless design. They also demonstrated awareness of security concerns by mentioning JWT authentication and rate limiting.",
    "System Design": "The candidate presented a good understanding of system design principles, particularly in optimizing a data pipeline. They reduced processing time by 70% through intelligent application of caching strategies and parallel processing."
  },
  "evaluation_summary": "The candidate demonstrates strong technical knowledge across all evaluated skills. Their Python expertise is particularly impressive with practical experience in multiple frameworks and performance optimization. While their REST API knowledge is solid, they could benefit from more examples of complex API design patterns. Overall, this is a strong candidate who would likely perform well in a Backend Software Engineer role."
}
```