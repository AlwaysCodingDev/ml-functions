import os
import uuid
import psycopg
import json
import base64
import jwt

# from validation import validate_job_description_or_modification

from typing import Annotated, Sequence, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, add_messages, START
from langgraph.checkpoint.postgres import PostgresSaver
import boto3
from botocore.exceptions import ClientError


def get_secret(key= "", jwt=False):
    secret_name = f"ecs/agent-example/{key}"
    region_name = "eu-west-2"

    if jwt:
        secret_name = f"amplify/jwt/secret"

    session = boto3.session.Session()
    client = session.client(service_name="secretsmanager", region_name=region_name)
    print("Secret name: ", secret_name)

    try:
        get_secret_value_response = client.get_secret_value(SecretId=secret_name)
    except ClientError as e:
        raise e

    secret = get_secret_value_response["SecretString"]

    if key == "postgres":
        return json.loads(secret)

    return secret


os.environ['OPENAI_API_KEY'] = get_secret("openai-key")
os.environ["JWT-Secret"] = get_secret(jwt=True)

DB_HOST = os.getenv("DB_RDS")
DB_PORT = int(os.getenv("DB_PORT"))
DB_USER = get_secret("postgres")['username']
DB_PASS = get_secret("postgres")['password']
DB_NAME = os.getenv("DB_NAME")

# CORS Headers
CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "Content-Type, Authorization"
}



class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


print("Initializing chat model")
model = init_chat_model("gpt-4.1-nano", model_provider="openai")

interview_prompt = """
You are an intelligent assistant that generates exactly 5 focused interview questions based strictly on a given job description.

Core Rules:

1. Use only the information provided in the job description.  
   - Do not assume, generalize, or add unstated requirements.
2. Focus solely on these components:
   - Job Title
   - Roles and Responsibilities
   - Required Skills and Qualifications
   - Experience Level

3. Question Quality Criteria:
   - Each question must be clear, specific, and professional.
   - Scope each question precisely to match the technical depth implied in the JD.
   - Questions must be of moderate difficulty, testing both understanding and application.
   - Avoid generic, vague, or high-level filler questions.
   - Ensure all questions are relevant, practical, and non-repetitive.

4. Generate exactly five questions, grouped by category:
   - Technical Skills (2)
   - Problem-Solving (1)
   - Behavioral (1)
   - Situational (1)

5. All questions must be grounded in the job description's content.

Additional Behavior:

- If the user requests a modification (e.g., rephrase, adjust difficulty, replace a question), retain the previous question set and modify only the specified part but return the whole modified part.
- Do not regenerate from scratch unless explicitly asked.

Output Guidelines:

- Return only the five interview questions.
- Always return all the five question if there are any modification done.
- Do not include any explanations, commentary, or section headers.
- Your output should be ready for immediate use in professional interview settings.
- Return the output as json string using question number as key and answer as value.
- Strictly follow all the instructions
- The output should stricitly follow json format
"""



prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system", interview_prompt
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)


def call_model(state: State):
    prompt = prompt_template.format_messages(**state)
    response = model.invoke(prompt)
    return {"messages": state["messages"] + [response]}


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
        decoded = jwt.decode(token, os.getenv("JWT-Secret"), algorithms=["HS256"])
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
            "body": json.dumps({"error token": str(e)}),
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
            "body": json.dumps({"error": f"Invalid JSON body: {str(e)}"})
        }

    query_user = data.get("query", "")
    jd = data.get("jd", "")
    thread_id = data.get("thread_id", "")

    if not query_user or not thread_id or not jd:
        return {
            "statusCode": 400,
            "headers": CORS_HEADERS,
            "body": json.dumps({"error": "Missing required fields: query, thread_id, jd"})
        }

    config = {"configurable": {"thread_id": thread_id}}
    query = f"User Query: {query_user}\n\nJob Description: {jd}"

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

            print("Chatting")
            input_messages = [HumanMessage(content=query)]
            result = app.invoke({"messages": input_messages}, config)
            print(f"Result: {result}")

            return {
                "statusCode": 200,
                "headers": CORS_HEADERS,
                "body": json.dumps({
                    "thread_id": thread_id,
                    "response": result["messages"][-1].content
                })
            }

    except Exception as e:
        return {
            "statusCode": 500,
            "headers": CORS_HEADERS,
            "body": json.dumps({"error body": str(e)})
        }