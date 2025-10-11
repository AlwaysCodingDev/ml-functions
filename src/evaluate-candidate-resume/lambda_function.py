import os
import boto3
import openai
import requests
import time
from botocore.exceptions import BotoCoreError, ClientError
from typing import TypedDict
from langgraph.graph import Graph, END
from keywordmatch import get_keywords_similarity_score
import json
import re


# Initialize logging client
logs_client = boto3.client('logs')


def put_log(group, stream, message):
    """Put log message to CloudWatch Logs"""
    ts = int(time.time() * 1000)
    try:
        logs_client.create_log_group(logGroupName=group)
    except logs_client.exceptions.ResourceAlreadyExistsException:
        pass

    try:
        logs_client.create_log_stream(logGroupName=group, logStreamName=stream)
    except logs_client.exceptions.ResourceAlreadyExistsException:
        pass

    logs_client.put_log_events(
        logGroupName=group,
        logStreamName=stream,
        logEvents=[{"timestamp": ts, "message": message}]
    )


def log_to_destinations(job_id, response_id, message):
    """Always log to central, conditionally to job/response."""
    put_log("areeva/central", "central", message)

    if job_id:
        put_log("areeva/jobs", f"job_{job_id}", message)

    if response_id:
        put_log("areeva/responses", f"response_{response_id}", message)


# def get_secret(key):
#     secret_name = f"ecs/agent-example/{key}"
#     region_name = "eu-west-2"

#     session = boto3.session.Session()
#     client = session.client(
#         service_name='secretsmanager',
#         region_name=region_name
#     )

#     try:
#         get_secret_value_response = client.get_secret_value(
#             SecretId=secret_name
#         )
#     except ClientError as e:
#         raise e

#     secret = get_secret_value_response['SecretString']
#     return secret


# os.environ['OPENAI_API_KEY'] = get_secret("openai-key")
# os.environ['API_TOKEN'] = get_secret("api_token")
# os.environ['API_KEY'] = get_secret("api_key")
dynamodb = boto3.resource("dynamodb")
s3 = boto3.client('s3')
openai.api_key = os.environ["OPENAI_API_KEY"]


class GraphState(TypedDict):
    """Simplified state schema for the LangGraph"""
    response_id: str
    job_id: str
    job_table: str
    response_table: str
    resumeText: str
    score: float
    error: str
    resume_cutoff: float
    is_valid_resume: bool


def validate_resume_text(state: GraphState) -> GraphState:
    """
    Node 0: Validate if the resume text is actually a valid resume
    """
    try:
        log_to_destinations(state["job_id"], state["response_id"], "Starting resume validation process")
        print("Validating resume text...")
        
        # Get candidate details from DynamoDB
        resp_table = dynamodb.Table(state["response_table"])
        resp_lookup = resp_table.get_item(Key={"id": state["response_id"]})
        resp_item = resp_lookup.get("Item")

        if not resp_item:
            error_msg = f"Response ID '{state['response_id']}' not found"
            log_to_destinations(state["job_id"], state["response_id"], f"ERROR: {error_msg}")
            state["error"] = error_msg
            return state

        state["resumeText"] = resp_item.get("resumeText", "").strip()
        if not state["resumeText"]:
            error_msg = "Missing or empty 'resumeText' for this response_id"
            log_to_destinations(state["job_id"], state["response_id"], f"ERROR: {error_msg}")
            state["error"] = error_msg
            return state

        log_to_destinations(state["job_id"], state["response_id"], f"Resume text retrieved successfully, length: {len(state['resumeText'])} characters")

        # Use OpenAI to validate if the text is actually a resume
        prompt = f"""
        Analyze the following text and determine if it is a valid resume/CV document.
        
        Text to analyze: {state['resumeText']}
        
        A valid resume should contain:
        - Personal information (name, contact details)
        - Work experience or employment history
        - Skills or qualifications
        - Education background
        - Professional summary or objective (optional)
        
        Respond with only "YES" if this is a valid resume, or "NO" if it's not a resume (e.g., if it's just random text, a cover letter only, incomplete information, etc.).
        """
        
        log_to_destinations(state["job_id"], state["response_id"], "Calling OpenAI for resume validation")
        completion = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
        )
        
        validation_result = completion.choices[0].message.content.strip().upper()
        state["is_valid_resume"] = validation_result == "YES"
        
        log_to_destinations(state["job_id"], state["response_id"], f"Resume validation result: {validation_result} (is_valid_resume: {state['is_valid_resume']})")
        
        if not state["is_valid_resume"]:
            log_to_destinations(state["job_id"], state["response_id"], "Invalid resume detected. Starting cleanup process...")
            print("Invalid resume detected. Cleaning up and setting scores to 0...")
            
            # Set all scores to 0
            zero_scores = {
                'role_similarity': '0',
                'education_similarity': '0',
                'skills_similarity': '0',
                'keywords_similarity': '0',
                'experience_similarity': '0',
                'certification_similarity': '0'
            }
            
            # Update response table with zero scores and mark interview completion
            resp_table.update_item(
                Key={"id": state["response_id"]},
                UpdateExpression="SET similarityScore = :score, detailedScores = :detailed, interviewCompletion[0] = :completion",
                ExpressionAttributeValues={
                    ":score": "0",
                    ":detailed": str(zero_scores),
                    ":completion": True
                }
            )
            log_to_destinations(state["job_id"], state["response_id"], "Updated response table with zero scores")
            
            # Get file information for S3 cleanup
            bucket_name = resp_item.get("bucket")
            filename = resp_item.get("fileName")
            resume = resp_item.get("resume")
            resume_name = resp_item.get("resumeName")
            candidate_id = resp_item.get("candidateId")
            
            # Remove file from S3 if it exists
            if bucket_name and filename:
                try:
                    s3.delete_object(Bucket=bucket_name, Key=filename)
                    log_to_destinations(state["job_id"], state["response_id"], f"Deleted file {filename} from S3 bucket {bucket_name}")
                    print(f"Deleted file {filename} from S3 bucket {bucket_name}")
                except Exception as s3_error:
                    error_msg = f"Error deleting S3 file: {s3_error}"
                    log_to_destinations(state["job_id"], state["response_id"], f"ERROR: {error_msg}")
                    print(error_msg)
            
            # Remove resume-related attributes from response table
            try:
                update_expression_parts = []
                expression_attribute_names = {}
                
                if bucket_name:
                    update_expression_parts.append("#bucket")
                    expression_attribute_names["#bucket"] = "bucket"
                
                if filename:
                    update_expression_parts.append("#fileName")
                    expression_attribute_names["#fileName"] = "fileName"
                
                if resume:
                    update_expression_parts.append("#resume")
                    expression_attribute_names["#resume"] = "resume"
                
                if resume_name:
                    update_expression_parts.append("#resumeName")
                    expression_attribute_names["#resumeName"] = "resumeName"
                
                if update_expression_parts:
                    resp_table.update_item(
                        Key={"id": state["response_id"]},
                        UpdateExpression=f"REMOVE {', '.join(update_expression_parts)}",
                        ExpressionAttributeNames=expression_attribute_names
                    )
                    log_to_destinations(state["job_id"], state["response_id"], "Removed resume-related attributes from response table")
                    print("Removed resume-related attributes from response table")
            except Exception as resp_error:
                error_msg = f"Error updating response table: {resp_error}"
                log_to_destinations(state["job_id"], state["response_id"], f"ERROR: {error_msg}")
                print(error_msg)
            
            # Update candidate table if candidate_id exists
            if candidate_id:
                try:
                    candidate_table_name = os.environ.get('CANDIDATE_TABLE_NAME', 'candidates')
                    candidate_table = dynamodb.Table(candidate_table_name)
                    
                    # Get current candidate data
                    candidate_resp = candidate_table.get_item(Key={"id": candidate_id})
                    candidate_item = candidate_resp.get("Item")
                    
                    if candidate_item:
                        current_resumes = candidate_item.get("resumes", [])
                        current_resume_names = candidate_item.get("resumeNames", [])
                        
                        # Remove the resume and resumeName from the lists
                        updated_resumes = [r for r in current_resumes if r != resume] if resume else current_resumes
                        updated_resume_names = [rn for rn in current_resume_names if rn != resume_name] if resume_name else current_resume_names
                        
                        # Update candidate table
                        candidate_table.update_item(
                            Key={"id": candidate_id},
                            UpdateExpression="SET resumes = :resumes, resumeNames = :resumeNames",
                            ExpressionAttributeValues={
                                ":resumes": updated_resumes,
                                ":resumeNames": updated_resume_names
                            }
                        )
                        log_to_destinations(state["job_id"], state["response_id"], f"Updated candidate {candidate_id} - removed resume references")
                        print(f"Updated candidate {candidate_id} - removed resume references")
                except Exception as cand_error:
                    error_msg = f"Error updating candidate table: {cand_error}"
                    log_to_destinations(state["job_id"], state["response_id"], f"ERROR: {error_msg}")
                    print(error_msg)
            
            state["score"] = 0.0
            log_to_destinations(state["job_id"], state["response_id"], "Resume validation failed. All cleanup completed.")
            print("Resume validation failed. All cleanup completed.")
        else:
            log_to_destinations(state["job_id"], state["response_id"], "Resume validation passed. Proceeding to next step.")
            print("Resume validation passed. Proceeding to next step.")
        
        return state

    except Exception as e:
        error_msg = f"[ERROR in validate_resume_text] {e}"
        log_to_destinations(state["job_id"], state["response_id"], f"ERROR: {error_msg}")
        print(error_msg)
        state["error"] = str(e)
        return state


def generate_tailored_questions(state: GraphState) -> GraphState:
    """
    Node 1: Generate tailored interview questions and insert into response_id doc
    """
    try:
        log_to_destinations(state["job_id"], state["response_id"], "Starting tailored questions generation")
        print("Generating tailored questions...")
        
        # Get job details from DynamoDB
        job_table = dynamodb.Table(state["job_table"])
        job_resp = job_table.get_item(Key={"id": state["job_id"]})
        job_item = job_resp.get("Item")

        if not job_item:
            error_msg = f"Job ID '{state['job_id']}' not found"
            log_to_destinations(state["job_id"], state["response_id"], f"ERROR: {error_msg}")
            state["error"] = error_msg
            return state

        log_to_destinations(state["job_id"], state["response_id"], "Job details retrieved successfully")

        # Generate tailored questions using OpenAI
        prompt = f"""
        You are an expert technical recruiter. Generate 10 highly tailored interview questions based on the candidate's resume and job requirements.

        Job Description: {job_item.get('jobDescriptionMarkdown', '')}
        Job Role: {job_item.get('jobRole', '')}
        Base Questions: {job_item.get('IQs', '')}
        Required Skills: {job_item.get('requiredSkills', [])}
        Education Requirements: {job_item.get('educationRequirements', '')}
        Years of Experience: {job_item.get('yearsOfExperience', '')}

        Candidate Resume: {state['resumeText']}

        Instructions:
        - Generate exactly 10 tailored questions
        - Use the Base Questions as reference and do not exactly copy the same
        - Focus on candidate's specific experience and skills
        - Align questions with job requirements
        - Be specific and avoid generic questions
        - Output only the numbered questions, nothing else
        """

        log_to_destinations(state["job_id"], state["response_id"], "Calling OpenAI for tailored questions generation")
        completion = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
        )
        
        tailored_questions = completion.choices[0].message.content.strip()
        log_to_destinations(state["job_id"], state["response_id"], f"Generated {len(tailored_questions.split(chr(10)))} tailored questions")
        
        # Insert tailored questions into response_id document
        resp_table = dynamodb.Table(state["response_table"])
        resp_table.update_item(
            Key={"id": state["response_id"]},
            UpdateExpression="SET TQs = :tq",
            ExpressionAttributeValues={":tq": tailored_questions}
        )

        log_to_destinations(state["job_id"], state["response_id"], "Tailored questions generated and stored successfully")
        print("Tailored questions generated and stored successfully")
        return state

    except Exception as e:
        error_msg = f"[ERROR in generate_tailored_questions] {e}"
        log_to_destinations(state["job_id"], state["response_id"], f"ERROR: {error_msg}")
        print(error_msg)
        state["error"] = str(e)
        return state


def calculate_similarity_scores(state: GraphState) -> GraphState:
    """
    Node 2: Calculate similarity scores for all job requirements
    """
    try:
        log_to_destinations(state["job_id"], state["response_id"], "Starting similarity scores calculation")
        print("Calculating similarity scores...")
        
        # Get job details
        job_table = dynamodb.Table(state["job_table"])
        job_resp = job_table.get_item(Key={"id": state["job_id"]})
        job_item = job_resp.get("Item")

        if not job_item:
            error_msg = f"Job ID '{state['job_id']}' not found"
            log_to_destinations(state["job_id"], state["response_id"], f"ERROR: {error_msg}")
            state["error"] = error_msg
            return state

        # Store resume cutoff for the next node
        state["resume_cutoff"] = float(job_item.get("resumeCutoff", 0.0))
        log_to_destinations(state["job_id"], state["response_id"], f"Resume cutoff threshold: {state['resume_cutoff']}")

        scores = {}
        
        # 1. Role Similarity Score
        log_to_destinations(state["job_id"], state["response_id"], "Calculating role similarity score")
        job_role = job_item.get('jobRole', '')
        role_score = get_role_similarity_score(state["resumeText"], job_role)
        scores['role_similarity'] = (str(role_score))
        log_to_destinations(state["job_id"], state["response_id"], f"Role similarity score: {role_score}")
        
        # 2. Education Requirements Score
        log_to_destinations(state["job_id"], state["response_id"], "Calculating education similarity score")
        education_requirements = job_item.get('educationRequirements', '')
        education_score = get_education_similarity_score(state["resumeText"], education_requirements)
        scores['education_similarity'] = (str(education_score))
        log_to_destinations(state["job_id"], state["response_id"], f"Education similarity score: {education_score}")
        
        # 3. Required Skills Score
        log_to_destinations(state["job_id"], state["response_id"], "Calculating skills similarity score")
        required_skills = job_item.get('requiredSkills', [])
        skills_score = get_skills_similarity_score(state["resumeText"], required_skills)
        scores['skills_similarity'] = (str(skills_score))
        log_to_destinations(state["job_id"], state["response_id"], f"Skills similarity score: {skills_score}")
        
        # 4. Keywords Score
        log_to_destinations(state["job_id"], state["response_id"], "Calculating keywords similarity score")
        keywords = job_item.get('keywords', [])
        keywords_score = get_keywords_similarity_score(state["resumeText"], keywords)
        scores['keywords_similarity'] = (str(round(keywords_score)))
        log_to_destinations(state["job_id"], state["response_id"], f"Keywords similarity score: {round(keywords_score)}")
        
        # 5. Years of Experience Score
        log_to_destinations(state["job_id"], state["response_id"], "Calculating experience similarity score")
        required_years = job_item.get('yearsOfExperience', 0)
        experience_score = get_experience_similarity_score(state["resumeText"], required_years)
        scores['experience_similarity'] = (str(experience_score))
        log_to_destinations(state["job_id"], state["response_id"], f"Experience similarity score: {experience_score}")
        
        # 6. Certification Score
        log_to_destinations(state["job_id"], state["response_id"], "Calculating certification similarity score")
        certification_score = get_certification_similarity_score(state["resumeText"], job_item)
        scores['certification_similarity'] = (str(certification_score))
        log_to_destinations(state["job_id"], state["response_id"], f"Certification similarity score: {certification_score}")
        
        # Calculate average similarity score
        valid_scores = [float(score) for score in scores.values() if score is not None]
        if valid_scores:
            final_score = sum(valid_scores) / len(valid_scores)
        else:
            final_score = 0.0
        
        state["score"] = final_score
        log_to_destinations(state["job_id"], state["response_id"], f"Final similarity score calculated: {final_score}")
        
        # Store similarity score in response_table
        resp_table = dynamodb.Table(state["response_table"])
        resp_table.update_item(
            Key={"id": state["response_id"]},
            UpdateExpression="SET similarityScore = :score, detailedScores = :detailed, interviewCompletion[0] = :completion",
            ExpressionAttributeValues={
                ":score": str(final_score),
                ":detailed": str(scores),
                ":completion": True
            }
        )
        
        log_to_destinations(state["job_id"], state["response_id"], f"Similarity scores stored in database. Final score: {final_score}, Detailed scores: {scores}")
        print(f"Similarity scores calculated. Final score: {final_score}")
        print(f"Detailed scores: {scores}")
        
        return state

    except Exception as e:
        error_msg = f"[ERROR in calculate_similarity_scores] {e}"
        log_to_destinations(state["job_id"], state["response_id"], f"ERROR: {error_msg}")
        print(error_msg)
        state["error"] = str(e)
        return state


def check_threshold_and_notify(state: GraphState) -> GraphState:
    """
    Node 3: Check if similarity score exceeds resume cutoff threshold and make API call
    """
    try:
        log_to_destinations(state["job_id"], state["response_id"], "Starting threshold check and notification process")
        print("Checking threshold and making API call...")
        
        resume_cutoff = state["resume_cutoff"]
        similarity_score = state["score"]
        
        log_to_destinations(state["job_id"], state["response_id"], f"Threshold comparison - Resume cutoff: {resume_cutoff}, Candidate score: {similarity_score}")
        print(f"Resume cutoff threshold: {resume_cutoff}")
        print(f"Candidate similarity score: {similarity_score}")
        
        if similarity_score >= resume_cutoff:
            log_to_destinations(state["job_id"], state["response_id"], "Candidate meets threshold requirements. Preparing API Gateway call...")
            print("Candidate meets threshold requirements. Making API Gateway call...")
            
            # Make HTTP call to API Gateway
            # You'll need to replace this URL with your actual API Gateway endpoint
            api_gateway_url = "https://ovuftgqb27.execute-api.eu-west-2.amazonaws.com/dev/api/v1/system/instant-resume-shortlist"
            
            payload = {
                "responseId": state["response_id"]
            }
            
            headers = {
                "Content-Type": "application/json"
            }

            log_to_destinations(state["job_id"], state["response_id"], f"Making API call to: {api_gateway_url} with payload: {payload}")
            print(type(api_gateway_url), payload)
            
            response = requests.post(
                api_gateway_url,
                json=payload,
                headers=headers,
                timeout=30
            )
            
            if response.status_code == 200:
                log_to_destinations(state["job_id"], state["response_id"], f"API Gateway call successful. Response: {response.json()}")
                print("API Gateway call successful")
                print(f"API Response: {response.json()}")
            else:
                error_msg = f"API Gateway call failed with status code: {response.status_code}, Response: {response.text}"
                log_to_destinations(state["job_id"], state["response_id"], f"ERROR: {error_msg}")
                print(f"API Gateway call failed with status code: {response.status_code}")
                print(f"Response: {response.text}")
                state["error"] = f"API Gateway call failed: {response.status_code} - {response.text}"
        else:
            log_message = f"Candidate does not meet threshold requirements. Score {similarity_score} < {resume_cutoff}"
            log_to_destinations(state["job_id"], state["response_id"], log_message)
            print(log_message)
        
        return state

    except Exception as e:
        error_msg = f"[ERROR in check_threshold_and_notify] {e}"
        log_to_destinations(state["job_id"], state["response_id"], f"ERROR: {error_msg}")
        print(error_msg)
        state["error"] = str(e)
        return state


def get_role_similarity_score(resume_text: str, job_role: str) -> float:
    """Calculate similarity between candidate's role and job role"""
    try:
        prompt = f"""
        Analyze the resume text and determine what role the candidate best fits (e.g., ML Engineer, Cloud Architect, Software Developer, etc.).
        
        Resume Text: {resume_text}
        Job Role Required: {job_role}
        
        Compare the candidate's role with the required job role and provide a similarity score from 0-100:
        - 100: Exactly the same role
        - 80-99: Very similar roles (same domain, similar responsibilities)
        - 60-79: Somewhat similar roles (overlapping skills)
        - 40-59: Different but related roles
        - 0-39: Completely different roles
        
        Return only the numerical score (0-100).
        """
        
        completion = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
        )
        
        score_text = completion.choices[0].message.content.strip()
        score = float(re.findall(r'\d+', score_text)[0])
        return min(100, max(0, score))
        
    except Exception as e:
        print(f"Error calculating role similarity: {e}")
        return 0.0


def get_education_similarity_score(resume_text: str, education_requirements: str) -> float:
    """Calculate similarity between candidate's education and job requirements"""
    try:
        prompt = f"""
        Compare the candidate's educational background with the job's education requirements.
        
        Resume Text: {resume_text}
        Education Requirements: {education_requirements}
        
        Provide a similarity score from 0-100:
        - 100: Exactly matches or exceeds requirements
        - 80-99: Very close match (same level, related field)
        - 60-79: Somewhat matches (same level, different field OR lower level, same field)
        - 40-59: Partially matches
        - 0-39: Does not match requirements
        
        Return only the numerical score (0-100).
        """
        
        completion = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
        )
        
        score_text = completion.choices[0].message.content.strip()
        score = float(re.findall(r'\d+', score_text)[0])
        return min(100, max(0, score))
        
    except Exception as e:
        print(f"Error calculating education similarity: {e}")
        return 0.0


def get_skills_similarity_score(resume_text: str, required_skills: list) -> float:
    """Calculate similarity between candidate's skills and required skills"""
    try:
        prompt = f"""
        Compare the candidate's skills with the required skills list.
        
        Resume Text: {resume_text}
        Required Skills: {required_skills}
        
        Analyze how many of the required skills the candidate has and at what level.
        Provide a similarity score from 0-100:
        - 100: Has all required skills at expert level
        - 80-99: Has most required skills at good level
        - 60-79: Has some required skills or most at basic level
        - 40-59: Has few required skills
        - 0-39: Has very few or no required skills
        
        Return only the numerical score (0-100).
        """
        
        completion = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
        )
        
        score_text = completion.choices[0].message.content.strip()
        score = float(re.findall(r'\d+', score_text)[0])
        return min(100, max(0, score))
        
    except Exception as e:
        print(f"Error calculating skills similarity: {e}")
        return 0.0


def get_experience_similarity_score(resume_text: str, required_years: int) -> float:
    """Calculate experience similarity score based on years difference"""
    try:
        prompt = f"""
        Extract the total years of professional experience from the resume.
        
        Resume Text: {resume_text}
        
        Look for work experience, employment history, and calculate total years.
        Return only the number of years as an integer.
        """
        
        completion = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
        )
        
        years_text = completion.choices[0].message.content.strip()
        candidate_years = int(re.findall(r'\d+', years_text)[0])
        
        # Calculate score based on difference
        difference = abs(candidate_years - required_years)
        
        if difference == 0:
            return 100.0
        elif difference <= 1:
            return 90.0
        elif difference <= 2:
            return 80.0
        elif difference <= 3:
            return 70.0
        elif difference <= 4:
            return 60.0
        elif difference <= 5:
            return 50.0
        else:
            return max(0.0, 50.0 - (difference - 5) * 10)
            
    except Exception as e:
        print(f"Error calculating experience similarity: {e}")
        return 0.0


def get_certification_similarity_score(resume_text: str, job_item: dict) -> float:
    """Calculate certification similarity score"""
    try:
        job_description = job_item.get('jobDescriptionMarkdown', '')
        
        prompt = f"""
        Extract certifications from the candidate's resume and compare with job requirements.
        
        Resume Text: {resume_text}
        Job Description: {job_description}
        
        Analyze if the candidate's certifications are relevant to the job requirements.
        Provide a similarity score from 0-100:
        - 100: Has all required certifications or highly relevant ones
        - 80-99: Has most required certifications or very relevant ones
        - 60-79: Has some relevant certifications
        - 40-59: Has few relevant certifications
        - 0-39: No relevant certifications
        
        Return only the numerical score (0-100).
        """
        
        completion = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
        )
        
        score_text = completion.choices[0].message.content.strip()
        score = float(re.findall(r'\d+', score_text)[0])
        return min(100, max(0, score))
        
    except Exception as e:
        print(f"Error calculating certification similarity: {e}")
        return 0.0


def should_continue(state: GraphState) -> str:
    """Router function to determine next step"""
    if state.get("error"):
        return "end"
    return "continue"


def should_process_resume(state: GraphState) -> str:
    """Router function to determine if resume should be processed further"""
    if state.get("error"):
        return "end"
    if not state.get("is_valid_resume", False):
        return "end"
    return "continue"


def create_workflow():
    """Create and configure the LangGraph workflow"""
    
    # Initialize the graph
    workflow = Graph()
    
    # Add nodes
    workflow.add_node("validate_resume", validate_resume_text)
    workflow.add_node("generate_questions", generate_tailored_questions)
    workflow.add_node("calculate_scores", calculate_similarity_scores)
    workflow.add_node("check_threshold", check_threshold_and_notify)
    
    # Set entry point
    workflow.set_entry_point("validate_resume")
    
    # Add edges
    workflow.add_conditional_edges(
        "validate_resume",
        should_process_resume,
        {
            "continue": "generate_questions",
            "end": END
        }
    )
    
    workflow.add_conditional_edges(
        "generate_questions",
        should_continue,
        {
            "continue": "calculate_scores",
            "end": END
        }
    )
    
    workflow.add_conditional_edges(
        "calculate_scores",
        should_continue,
        {
            "continue": "check_threshold",
            "end": END
        }
    )
    
    workflow.add_edge("check_threshold", END)
    
    return workflow.compile()


def lambda_handler(event, context):
    """Lambda handler that orchestrates the streamlined workflow"""
    
    log_to_destinations(event.get("job_id"), event.get("response_id"), f"Lambda execution started with event: {json.dumps(event, default=str)}")
    
    # Validate required fields
    required_keys = ["response_id", "job_id", "job_table", "response_table"]
    missing = [k for k in required_keys if k not in event]
    if missing:
        error_msg = f"Missing required field(s): {', '.join(missing)}"
        log_to_destinations(event.get("job_id"), event.get("response_id"), f"ERROR: {error_msg}")
        return {
            "error": error_msg
        }
    
    # Initialize state
    initial_state = GraphState(
        response_id=event["response_id"],
        job_id=event["job_id"],
        job_table=event["job_table"],
        response_table=event["response_table"],
        resumeText="",
        score=0.0,
        error="",
        resume_cutoff=0.0,
        is_valid_resume=False
    )
    
    log_to_destinations(event["job_id"], event["response_id"], f"Initialized workflow state for response_id: {event['response_id']}, job_id: {event['job_id']}")
    
    try:
        # Create and run the workflow
        log_to_destinations(event["job_id"], event["response_id"], "Creating and starting workflow execution")
        workflow = create_workflow()
        result = workflow.invoke(initial_state)
        
        # Check for errors
        if result.get("error"):
            log_to_destinations(event["job_id"], event["response_id"], f"Workflow completed with error: {result['error']}")
            return {"error": result["error"]}
        
        
        # Return success response
        res = {
            "message": "Candidate analysis workflow completed successfully.",
            "response_id": result["response_id"],
            "similarity_score": result["score"],
            "resume_cutoff": result.get("resume_cutoff", 0.0),
            "meets_threshold": result["score"] >= result.get("resume_cutoff", 0.0),
            "is_valid_resume": result.get("is_valid_resume", False),
            "tailored_questions_generated": result.get("is_valid_resume", False),
            "workflow_completed": True
        }

        log_to_destinations(event["job_id"], event["response_id"], f"Workflow completed successfully. Final result: {json.dumps(res, default=str)}")
        print(f"[SUCCESS] {res}")

        return res
        
    except (ClientError, BotoCoreError) as aws_err:
        error_msg = f"[AWS/DynamoDB error] {aws_err}"
        log_to_destinations(event.get("job_id"), event.get("response_id"), f"ERROR: {error_msg}")
        print(error_msg)
        return {"error": f"AWS/DynamoDB error: {str(aws_err)}"}
    
    except Exception as e:
        error_msg = f"[ERROR] {e}"
        log_to_destinations(event.get("job_id"), event.get("response_id"), f"ERROR: {error_msg}")
        print(error_msg)
        return {"error": str(e)}