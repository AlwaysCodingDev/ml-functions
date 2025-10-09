import os
import psycopg
from typing import Annotated, Sequence, TypedDict
import json
import base64

# from validation import validate_job_description_or_modification
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, add_messages, START
from langgraph.checkpoint.postgres import PostgresSaver

import boto3
from botocore.exceptions import ClientError
import jwt

# Get secrets
# def get_secret(key= "", jwt=False):
#     secret_name = f"ecs/agent-example/{key}"
#     region_name = "eu-west-2"

#     if jwt:
#         secret_name = f"amplify/jwt/secret"

#     session = boto3.session.Session()
#     client = session.client(service_name="secretsmanager", region_name=region_name)
#     print("Secret name: ", secret_name)

#     try:
#         get_secret_value_response = client.get_secret_value(SecretId=secret_name)
#     except ClientError as e:
#         raise e

#     secret = get_secret_value_response["SecretString"]

#     if key == "postgres":
#         return json.loads(secret)

#     return secret


# os.environ["OPENAI_API_KEY"] = get_secret("openai-key")
# os.environ["JWT_Secret"] = get_secret(jwt=True)

DB_HOST = os.getenv("DB_RDS")
DB_PORT = int(os.getenv("DB_PORT"))
# DB_USER = get_secret("postgres")["username"]
# DB_PASS = get_secret("postgres")["password"]
DB_USER=os.getenv("username")
DB_PASS=os.getenv("password")
DB_NAME = os.getenv("DB_NAME")

# CORS headers for proxy integration
CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",  # Use specific domain in prod if needed
    "Access-Control-Allow-Headers": "Content-Type, Authorization",
}


# Define State schema
class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


# Init model
print("Initializing chat model")
model = init_chat_model("gpt-4.1-nano", model_provider="openai")

# Prompt Template
prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                """You are an intelligent assistant that generates professional, detailed job descriptions ready for direct social media posting, job boards, and hiring platforms.

Follow these strict rules when formatting your output:

## Mandatory Output Format
Always return only the in valid Markdown, clean and ready for copy-paste.  
Do not include commentary, explanations, or metadata.  

The job description must include exactly and only the following components if present in the input:  
- Job Title  
- Company Name (if available)  
- Location  
- Employment Type  
- Experience Level  
- Roles and Responsibilities  
- Required Skills and Qualifications  
- Preferred/Bonus Skills (explain in detail using bullet points)  
- Education Requirements 

## Writing Guidelines
- Job descriptions must always be **legal, valid, and industry-appropriate**.  
- Word count must be **exactly 300 words**.  
- Do not infer or invent company-specific, compensation, culture, or benefit details unless explicitly provided.  
- Do not include emojis, hashtags, or irrelevant content.  
- Use headers and bullet points for clarity.  
- Add standard, role-aligned responsibilities or skills only when implied by the role, and clearly mark them as *typical* or *expected*.  
- Ensure content is logically structured, precise, and free of repetition or filler.  

## Operational Principle
- Always prioritize correctness and consistency.  
- Maintain the **last valid job description** as a safe fallback.  
- Support incremental updates from the user without breaking formatting or word-count rules."""
            ),
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)


# Model node
def call_model(state: State):
    prompt = prompt_template.format_messages(**state)
    response = model.invoke(prompt)
    return {"messages": state["messages"] + [response]}


# Lambda handler
def lambda_handler(event, context):
    
    client_id = None

    try:
        header = event.get("headers", {})
        auth_header = header.get("Authorization", "")  # Get the Authorization header

        if not auth_header:
            raise ValueError("Authorization header missing")

        # Extract the token from "Bearer <token>"
        parts = auth_header.split()
        if len(parts) != 2 or parts[0].lower() != "bearer":
            raise ValueError("Invalid Authorization header format")

        token = parts[1]

        # Decode JWT
        decoded = jwt.decode(token, os.getenv("JWT_Secret"), algorithms=["HS256"])
        client_id = decoded.get("clientId", None)

        if not client_id:
            raise ValueError("Client ID not found in token")

    except jwt.ExpiredSignatureError:
        return {
            "statusCode": 401,
            "headers": CORS_HEADERS,
            "body": json.dumps({"error": "Token has expired"}),
        }

    except jwt.InvalidTokenError:
        return {
            "statusCode": 401,
            "headers": CORS_HEADERS,
            "body": json.dumps({"error": "Invalid token"}),
        }

    except Exception as e:
        return {
            "statusCode": 401,
            "headers": CORS_HEADERS,
            "body": json.dumps({"error": str(e)}),
        }

    try:
        body = event.get("body", "")
        if event.get("isBase64Encoded"):
            body = base64.b64decode(body).decode()
        data = json.loads(body or "{}")
    except Exception as e:
        return {
            "statusCode": 400,
            "headers": CORS_HEADERS,
            "body": json.dumps({"error": f"Invalid JSON body: {str(e)}"}),
        }

    query = data.get("query")
    thread_id = data.get("thread_id", "")

    if not query or not thread_id:
        return {
            "statusCode": 400,
            "headers": CORS_HEADERS,
            "body": json.dumps({"error": "Missing required fields: query, thread_id"}),
        }

    print(f"Received query: {query}")
    print(f"Received thread_id: {thread_id}")
    config = {"configurable": {"thread_id": thread_id}}

    # if validate_job_description_or_modification(query) is None:

    #     print("Invalid job description")
    #     return {
    #         "statusCode": 400,
    #         "headers": CORS_HEADERS,
    #         "body": json.dumps({"error": "Invalid job description"})
    #     }

    try:
        with psycopg.connect(
            host=DB_HOST, port=DB_PORT, user=DB_USER, password=DB_PASS, dbname=DB_NAME
        ) as conn:
            print("Connected to database")
            conn.autocommit = True
            saver = PostgresSaver(conn)
            saver.setup()

            workflow = StateGraph(state_schema=State)
            workflow.add_edge(START, "model")
            workflow.add_node("model", call_model)
            app = workflow.compile(checkpointer=saver)
            print("Workflow compiled")

            input_messages = [HumanMessage(content=query)]
            result = app.invoke({"messages": input_messages}, config)

            print("Job description generated")
            print(result["messages"][-1].content)

            return {
                "statusCode": 200,
                "headers": CORS_HEADERS,
                "body": json.dumps(
                    {"thread_id": thread_id, "response": result["messages"][-1].content}
                ),
            }

    except Exception as e:
        print(f"Error: {e}")
        return {
            "statusCode": 500,
            "headers": CORS_HEADERS,
            "body": json.dumps({"error": str(e)}),
        }
