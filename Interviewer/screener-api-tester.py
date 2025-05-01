import httpx
import os

def test_check_eligibility():
    url = "http://127.0.0.1:8000/check-eligibility/"
    resume_file_path = "C:\\Users\\zuhai\\Desktop\\resume.pdf"
    
    # Log file details before opening
    print(f"TESTER: Checking if file exists: {os.path.exists(resume_file_path)}")
    print(f"TESTER: File size: {os.path.getsize(resume_file_path) if os.path.exists(resume_file_path) else 'N/A'} bytes")
    
    job_description = """ Job description
About Afiniti

At Afiniti, we are a leading provider of artificial intelligence technology that elevates the customer experience by making moments of human connection more valuable. Our mission is rooted in a simple yet powerful idea: understanding patterns of human behavior enables us to predict how people will interact and create meaningful connections.

Using our patented AI technology, we revolutionize the contact center industry by pairing customers with the most compatible contact center agents. By doing so, we enhance the entire customer journey, resulting in exceptional experiences and improved outcomes for all parties involved.

Our transformative technology has generated billions of dollars in incremental value for our esteemed clients, which include Fortune 500 companies across diverse industries such as financial services, telecommunications, travel, and hospitality. We take pride in our global reach and impact, with our solutions being leveraged by organizations around the world.

To learn more about Afiniti and the groundbreaking work we do, visit www.afiniti.com.

About the role

Data Scientist I - Pakistan

Afiniti uses data science to enhance human interactions in large enterprises by efficiently pairing customers or tasks with company representatives. Our primary focus is improving contact center interactions for sales, service, retention, collections, and customer satisfaction in fields ranging from telecommunications to healthcare to banking to hospitality. To ensure we're delivering value, we measure performance using a real-time control group – routing a portion of calls using the client's existing system and the majority of calls using our data-scientist-designed pairing and next-best-action recommendation strategies.

Key Responsibilities
- Perform statistical inference on large data sets to inform decisions and drive actions.
- Drill down on results (problem-solving analysis) and conduct custom analysis.
- Build predictive models to optimize agent and customer interactions
- Collaborate and communicate with data analytics, and cross-functional teams on a task basis.
- Adapt theoretical concepts and standardized techniques to real-world problems
- Working at a stretch in CET hours as per client requirements (starting work post midday till midnight).
- Lead production activities on deployed accounts, or execute on deployment plan.

Minimum Qualifications
- Bachelors or Masters in Computer Science, Mathematics, Economics, Physics, Engineering or related quantitative field.
- 0-2 years of working experience in relevant field.
- Programming experience in one or more of the following languages i.e. R/Python/ Julia/SQL.
- Experience with statistics, machine learning, linear programming, or mathematical optimization, both practical and theoretical
- Strong attention to details.

Preferred Qualifications
- Excellent skills at distilling complex, ambiguous scenarios into tractable models
- Proficiency in R and Python
- Experience with data analysis, data visualization, programming, processing large data sets
- Familiarity with SQL, relational databases, version control, and tools for reproducibility such as git, Jupyter and R Markdown, make, or authoring custom packages
- Demonstrated ability to manage time independently and take projects to completion
- Willingness to learn new techniques
- Ability to document and explain cutting-edge techniques to other team members
- Comfort working in a collaborative environment with cross-team communication to bring projects into production
- Familiarity with Bayesian statistics, hierarchical modeling, MCMC algorithms, latent factor models

Location/Remote work statement

This is a hybrid opportunity required to work in CET Hours. """

    print(f"TESTER: Job description length: {len(job_description)} characters")

    try:
        # Open the resume file in binary mode
        with open(resume_file_path, "rb") as resume_file:
            # Get file size
            resume_file.seek(0, 2)  # Go to end of file
            file_size = resume_file.tell()
            resume_file.seek(0)  # Go back to beginning
            print(f"TESTER: Opened file successfully, size: {file_size} bytes")
            
            print("TESTER: Sending POST request to API...")
            # Send a POST request to the API
            response = httpx.post(
                url,
                files={"resume": ("sample_resume.pdf", resume_file, "application/pdf")},
                data={"job_description": job_description},
                timeout=60.0  # Add timeout to prevent hanging indefinitely
            )
            
            # Print the response
            print(f"TESTER: Status Code: {response.status_code}")
            
            try:
                json_response = response.json()
                print(f"TESTER: Response JSON: {json_response}")
            except Exception as e:
                print(f"TESTER: Failed to parse JSON response: {str(e)}")
                print(f"TESTER: Raw response: {response.text}")
    
    except FileNotFoundError:
        print(f"TESTER ERROR: Could not find file at path: {resume_file_path}")
    except Exception as e:
        print(f"TESTER ERROR: Unexpected error: {str(e)}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    test_check_eligibility()