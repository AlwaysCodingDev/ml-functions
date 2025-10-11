import os
# import psycopg
from typing import Annotated, Sequence, TypedDict
import json
import base64

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, add_messages, START
from langgraph.checkpoint.redis import RedisSaver

import jwt
import logging



logger = logging.getLogger()
handler = logging.StreamHandler()
formatter = logging.Formatter(
    "%(asctime)s - %(pathname)s - %(name)s - %(lineno)d - %(funcName)s- %(levelname)s  - %(message)s"
)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


DB_HOST = os.getenv("DB_RDS")
DB_PORT = int(os.getenv("DB_PORT"))
DB_USER=os.getenv("username")
DB_PASS=os.getenv("password")
DB_NAME = os.getenv("DB_NAME")
MODEL_NAME = os.getenv("MODEL_NAME")
MODEL_PROVIDER = os.getenv("MODEL_PROVIDER")

# CORS headers for proxy integration
CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",  # Use specific domain in prod if needed
    "Access-Control-Allow-Headers": "Content-Type, Authorization",
}


# Define State schema
class State(TypedDict):
    """
    State schema for the chat model
    """
    messages: Annotated[Sequence[BaseMessage], add_messages]

# Init model
logger.info("Initializing chat model")
model = init_chat_model(MODEL_NAME, model_provider=MODEL_PROVIDER)
logger.info("Chat model initialized")

# Prompt Template
logger.info("Creating prompt template")
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
    """
    Model node for the chat model
    """
    try:
        logger.info("Calling model")
        prompt = prompt_template.format_messages(**state)
        response = model.invoke(prompt)
        return {"messages": state["messages"] + [response]}
    except Exception as e:
        logger.error("Exception in model node", exc_info=True)
        raise

# Lambda handler
def lambda_handler(event, context):
    """
    Lambda handler for the chat model
    """
    client_id = None

    try:
        logger.info("Lambda handler started")
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
        logger.error("Token has expired", exc_info=True)
        return {
            "statusCode": 401,
            "headers": CORS_HEADERS,
            "body": json.dumps({"error": "Token has expired"}),
        }

    except jwt.InvalidTokenError:
        logger.error("Invalid token", exc_info=True)
        return {
            "statusCode": 401,
            "headers": CORS_HEADERS,
            "body": json.dumps({"error": "Invalid token"}),
        }

    except Exception as e:
        logger.error("Exception in lambda handler", exc_info=True)
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
        logger.error("Exception in lambda handler", exc_info=True)
        return {
            "statusCode": 400,
            "headers": CORS_HEADERS,
            "body": json.dumps({"error": f"Invalid JSON body: {str(e)}"}),
        }
    
    query = data.get("query")
    thread_id = data.get("thread_id", "")

    if not query or not thread_id:
        logger.error("Missing required fields: query, thread_id")
        return {
            "statusCode": 400,
            "headers": CORS_HEADERS,
            "body": json.dumps({"error": "Missing required fields: query, thread_id"}),
        }

    logger.info("Received query: %s", query)
    logger.info("Received thread_id: %s", thread_id)
    config = {"configurable": {"thread_id": thread_id}}

    try:
        with RedisSaver.from_conn_string(os.getenv("REDIS_URI")) as checkpointer:
            logger.info("Connected to database")
            checkpointer.setup() 
            workflow = StateGraph(state_schema=State)
            workflow.add_edge(START, "model")
            workflow.add_node("model", call_model)
            app = workflow.compile(checkpointer=checkpointer)
            logger.info("Workflow compiled")

            input_messages = [HumanMessage(content=query)]
            logger.debug("Input messages: %s", input_messages)
            logger.debug("Config: %s", config)
            result = app.invoke({"messages": input_messages}, config)
            logger.debug("Result: %s", result)

            logger.info("Job description generated")
            logger.debug(result["messages"][-1].content)

            return {
                "statusCode": 200,
                "headers": CORS_HEADERS,
                "body": json.dumps(
                    {"thread_id": thread_id, "response": result["messages"][-1].content}
                ),
            }

    except Exception as e:
        logger.error("Exception in lambda handler", exc_info=True)
        return {
            "statusCode": 500,
            "headers": CORS_HEADERS,
            "body": json.dumps({"error": str(e)}),
        }
