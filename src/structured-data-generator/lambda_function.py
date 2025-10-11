import os
from typing import Annotated, Sequence, TypedDict
import json
import base64
import logging
import sys
import jwt

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, add_messages, START
from langgraph.checkpoint.redis import RedisSaver

# import boto3
# from botocore.exceptions import ClientError

# ----------------------------
# Logging setup (structured-ish)
# ------------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)

_handler = logging.StreamHandler(sys.stdout)
_formatter = logging.Formatter(
    "%(asctime)s - %(pathname)s - %(name)s - %(lineno)d - %(funcName)s- %(levelname)s  - %(message)s"
)
_handler.setFormatter(_formatter)


# Prevent duplicate handlers if Lambda reuses the runtime
if not logger.handlers:
    logger.addHandler(_handler)
else:
    # Replace existing handler formatter/level if already present
    logger.handlers[0].setLevel(LOG_LEVEL)
    logger.handlers[0].setFormatter(_formatter)

def log_event(level: str, **kwargs):
    """Emit a JSON-ish line while staying simple/fast for Lambda."""
    msg = json.dumps(kwargs, default=str)
    logger.log(getattr(logging, level.upper(), logging.INFO), msg)

# ----------------------------
# Secrets
# ----------------------------
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

# Environment / config
# os.environ['OPENAI_API_KEY'] = get_secret("openai-key")
# os.environ["JWT-Secret"] = get_secret(jwt=True)

DB_HOST = os.getenv("DB_RDS")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
# DB_USER = get_secret("postgres")['username']
# DB_PASS = get_secret("postgres")['password']
DB_USER=os.getenv("username")
DB_PASS=os.getenv("password")
DB_NAME = os.getenv("DB_NAME")

MODEL_NAME = os.getenv("MODEL_NAME")
MODEL_PROVIDER = os.getenv("MODEL_PROVIDER")

# CORS headers for proxy integration
CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",  # Use specific domain in prod if needed
    "Access-Control-Allow-Headers": "Content-Type, Authorization"
}

# ----------------------------
# LangGraph / Model wiring
# ----------------------------
class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# Init model
logger.info("Initializing chat model")
logger.info("Model Name: %s", os.getenv("MODEL_NAME"))
logger.info("Model Provider: %s", os.getenv("MODEL_PROVIDER"))
model = init_chat_model(MODEL_NAME, model_provider=MODEL_PROVIDER)

# Prompt Template
prompt_template = ChatPromptTemplate.from_messages([
    ("system", (
        """
        You are a structured information extraction agent.  
        Given a Job Description (JD), your task is to extract only the information that is explicitly present in the description. Do not make assumptions or infer any details. Return the results in a json structured format.

        Follow these guidelines:
        - Analyze the JD and extract only the fields mentioned. Leave all others blank or empty.
        - Return your response in the following structured format with keys and appropriate values:
        - Don't use any special characters while answering, only normal plain text no special character like hypen etc.
        - Return the answer in json string format
        - Don't use newline characters
        - Always start editing the last output in the memory if available instead of the provided job description

        Structured Output Format:
        - Role Title:
        - Company:
        - Work Experience Required:
        - Employment Type (e.g., Full-time, Part-time, Contract):
        - Work Location (Remote/On-site/Hybrid, City, Country):
        - Skills Required (list of hard skills mentioned):
        - Languages Known (e.g., Python, JavaScript, C++, etc.):
        - Education Requirements:
        - Certifications (if any):
        - Projects or Domain Experience:
        - Project Description:
        - Technologies Used:
        - Soft Skills (e.g., communication, leadership):
        - Job Responsibilities:
        - Other Requirements (e.g., travel readiness, shift flexibility):

        Additional Instructions:
        - Do NOT fill in any field that is NOT explicitly stated in the job description.
        - Use bullet points or lists where applicable.
        - If a skill or requirement is labeled as "preferred" or "optional", mark it accordingly within the relevant section or include it in "Other Requirements".
        - Maintain concise and clear phrasing.
        - After presenting the structured response, allow the user to manually fine-tune or add missing information.
        - Return JSON format, follow it strictly.
        """
    )),
    MessagesPlaceholder(variable_name="messages"),
])

def _extract_token_usage_from_response(response) -> dict:
    """
    Try multiple likely locations for token usage depending on langchain/openai versions.
    Returns a dict with prompt_tokens, completion_tokens, total_tokens when available.
    """
    logger.info("Extracting token usage from response")
    usage = {}

    # LangChain's BaseMessage / AIMessage often carries response_metadata
    try:
        meta = getattr(response, "response_metadata", None) or {}
        # Common LC convention
        if isinstance(meta, dict):
            # e.g., meta.get("token_usage") or meta.get("usage")
            token_usage = meta.get("token_usage") or meta.get("usage") or {}
            if token_usage:
                usage.update({
                    "prompt_tokens": token_usage.get("prompt_tokens"),
                    "completion_tokens": token_usage.get("completion_tokens"),
                    "total_tokens": token_usage.get("total_tokens"),
                })
        # Some providers stash the raw OpenAI usage under additional_kwargs
        addl = getattr(response, "additional_kwargs", None) or {}
        if isinstance(addl, dict):
            raw_usage = addl.get("usage") or {}
            if raw_usage:
                usage.update({
                    "prompt_tokens": raw_usage.get("prompt_tokens"),
                    "completion_tokens": raw_usage.get("completion_tokens"),
                    "total_tokens": raw_usage.get("total_tokens"),
                })
    except Exception:
        # Never fail the request due to metrics extraction
        pass

    # Final safety net: try dict() if supported
    if not usage:
        try:
            as_dict = response.dict()
            raw_usage = (
                as_dict.get("response_metadata", {}).get("token_usage") or
                as_dict.get("response_metadata", {}).get("usage") or
                as_dict.get("additional_kwargs", {}).get("usage") or {}
            )
            if raw_usage:
                usage.update({
                    "prompt_tokens": raw_usage.get("prompt_tokens"),
                    "completion_tokens": raw_usage.get("completion_tokens"),
                    "total_tokens": raw_usage.get("total_tokens"),
                })
        except Exception:
            pass

    # Remove Nones for cleaner logs
    usage = {k: v for k, v in usage.items() if v is not None}
    return usage

def call_model(state: State, thread_id: str):
    prompt = prompt_template.format_messages(**state)
    response = model.invoke(prompt)

    # Token usage logging
    usage = _extract_token_usage_from_response(response)
    log_event(
        "info",
        event="model_usage",
        thread_id=thread_id,
        **({"prompt_tokens": usage.get("prompt_tokens")} if "prompt_tokens" in usage else {}),
        **({"completion_tokens": usage.get("completion_tokens")} if "completion_tokens" in usage else {}),
        **({"total_tokens": usage.get("total_tokens")} if "total_tokens" in usage else {})
    )

    return {"messages": state["messages"] + [response]}

# ----------------------------
# Lambda handler
# ----------------------------
def lambda_handler(event, context):
    # Log basic request context (avoid logging bodies/secrets)
    request_id = getattr(context, "aws_request_id", None)
    
    # DEBUG: Log the entire event structure (safely)
    try:
        safe_event = {k: v if k not in ['body'] else '<BODY_CONTENT>' for k, v in (event or {}).items()}
        log_event("debug", event="event_structure", safe_event=safe_event, request_id=request_id)
    except Exception as debug_e:
        log_event("error", event="debug_logging_failed", error=str(debug_e), request_id=request_id)
    
    log_event("info", event="request_received", request_id=request_id)

    # Handle preflight CORS
    if event and event.get("httpMethod") == "OPTIONS":
        return {"statusCode": 200, "headers": CORS_HEADERS, "body": ""}

    client_id = None

    try:
        # FIX 1: Add extensive null checks for headers
        if not event:
            raise ValueError("Event is None")
        
        headers = event.get("headers")
        if headers is None:
            # Try alternative header locations
            headers = event.get("Headers") or event.get("multiValueHeaders") or {}
        
        log_event("debug", event="headers_debug", headers_type=type(headers).__name__, 
                 headers_keys=list(headers.keys()) if isinstance(headers, dict) else "NOT_DICT", 
                 request_id=request_id)
        
        if not isinstance(headers, dict):
            raise ValueError(f"Headers is not a dict, got {type(headers)}: {headers}")
        
        auth_header = headers.get("Authorization") or headers.get("authorization", "")  # Try both cases

        if not auth_header:
            raise ValueError("Authorization header missing")

        # Extract the token from "Bearer <token>"
        parts = auth_header.split()
        if len(parts) != 2 or parts[0].lower() != "bearer":
            raise ValueError("Invalid Authorization header format")

        token = parts[1]

        # Decode JWT - FIX 2: Add null check for JWT secret
        jwt_secret = os.getenv("JWT_Secret")
        if not jwt_secret:
            raise ValueError("JWT secret not configured")
        
        log_event("debug", event="jwt_decode_attempt", request_id=request_id)
        decoded = jwt.decode(token, jwt_secret, algorithms=["HS256"])
        
        if not isinstance(decoded, dict):
            raise ValueError(f"JWT decode returned non-dict: {type(decoded)}")
        
        client_id = decoded.get("clientId", None)
        log_event("debug", event="jwt_decoded", client_id=client_id, request_id=request_id)

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

    # Parse body safely - FIX 3: Better null handling for body
    try:
        log_event("debug", event="body_parsing_start", request_id=request_id)
        
        if not event:
            raise ValueError("Event is None during body parsing")
        
        body = event.get("body")
        log_event("debug", event="body_raw", body_type=type(body).__name__, 
                 body_length=len(body) if body else 0, request_id=request_id)
        
        if body is None:
            body = ""
        
        if event.get("isBase64Encoded"):
            body = base64.b64decode(body).decode()
        
        # Handle empty body case
        if not body.strip():
            data = {}
        else:
            data = json.loads(body)
        
        log_event("debug", event="body_parsed", data_keys=list(data.keys()) if isinstance(data, dict) else "NOT_DICT", 
                 request_id=request_id)
                 
    except Exception as e:
        log_event("error", event="invalid_json", error=str(e), error_type=type(e).__name__, request_id=request_id)
        return {
            "statusCode": 400,
            "headers": CORS_HEADERS,
            "body": json.dumps({"error": f"Invalid JSON body: {str(e)}"})
        }

    # FIX 4: Better validation with default values
    try:
        log_event("debug", event="data_extraction_start", data_type=type(data).__name__, request_id=request_id)
        
        if not isinstance(data, dict):
            raise ValueError(f"Parsed data is not a dict: {type(data)}")
        
        query = data.get("query", "").strip() if data.get("query") else ""
        thread_id = data.get("thread_id", "").strip() if data.get("thread_id") else ""
        jd = data.get("jd", "").strip() if data.get("jd") else ""
        
        log_event("debug", event="data_extracted", 
                 query_len=len(query), thread_id_len=len(thread_id), jd_len=len(jd), 
                 request_id=request_id)
                 
    except Exception as e:
        log_event("error", event="data_extraction_error", error=str(e), error_type=type(e).__name__, request_id=request_id)
        return {
            "statusCode": 400,
            "headers": CORS_HEADERS,
            "body": json.dumps({"error": f"Data extraction error: {str(e)}"})
        }

    # Validate inputs
    if not query or not thread_id or not jd:
        log_event(
            "warning",
            event="missing_params",
            request_id=request_id,
            has_query=bool(query),
            has_thread_id=bool(thread_id),
            has_jd=bool(jd),
        )
        return {
            "statusCode": 400,
            "headers": CORS_HEADERS,
            "body": json.dumps({"error": "Missing required fields: query, thread_id, jd"})
        }

    log_event(
        "info",
        event="inputs_received",
        request_id=request_id,
        thread_id=thread_id
    )

    # FIX: Proper config setup for LangGraph checkpointer
    config = {
        "configurable": {
            "thread_id": thread_id,
            "checkpoint_ns": "",  # Add checkpoint namespace
        }
    }

    try:
        # FIX 5: Add validation for DB connection parameters
        # log_event("debug", event="db_config_check", 
        #          has_host=bool(DB_HOST), has_user=bool(DB_USER), 
        #          has_pass=bool(DB_PASS), has_name=bool(DB_NAME), 
        #          request_id=request_id)
        
        # if not all([DB_HOST, DB_USER, DB_PASS, DB_NAME]):
        #     raise ValueError("Database configuration incomplete")
        
        # Initialize checkpointer
        with RedisSaver.from_conn_string(os.getenv("REDIS_URI")) as checkpointer:
            logger.info("Connected to database")
            checkpointer.setup() 
        
            log_event("debug", event="workflow_setup", request_id=request_id)
            workflow = StateGraph(state_schema=State)
            workflow.add_edge(START, "model")
            # Wrap our model node so we can pass thread_id for logging
            workflow.add_node("model", lambda s: call_model(s, thread_id))
            app = workflow.compile(checkpointer=saver)

            log_event("info", event="workflow_compiled", request_id=request_id)

            input_messages = [HumanMessage(content=f"Job Description: {jd}\n{query}")]
            log_event("debug", event="model_invoke_start", request_id=request_id)
            
            # FIX: Handle checkpointer issues with retry logic
            max_retries = 2
            for attempt in range(max_retries):
                try:
                    result = app.invoke({"messages": input_messages}, config)
                    break  # Success, exit retry loop
                except TypeError as te:
                    if "'NoneType' object is not a mapping" in str(te) and attempt < max_retries - 1:
                        log_event("warning", event="checkpoint_corruption_detected", 
                                    attempt=attempt, thread_id=thread_id, request_id=request_id)
                        
                        # Clear the corrupted checkpoint and try again
                        try:
                            # Generate a new thread_id for this retry
                            new_thread_id = f"{thread_id}_retry_{attempt + 1}"
                            config["configurable"]["thread_id"] = new_thread_id
                            log_event("info", event="using_new_thread_id", 
                                        old_thread_id=thread_id, new_thread_id=new_thread_id, 
                                        request_id=request_id)
                            continue  # Try again with new thread_id
                        except Exception as cleanup_e:
                            log_event("error", event="checkpoint_cleanup_failed", 
                                        error=str(cleanup_e), request_id=request_id)
                    raise  # Re-raise if not a checkpointer issue or max retries exceeded
            
            log_event("debug", event="model_invoke_complete", 
                        result_type=type(result).__name__, 
                        result_keys=list(result.keys()) if isinstance(result, dict) else "NOT_DICT",
                        request_id=request_id)

            # FIX 6: Validate result structure before accessing
            if not result:
                raise ValueError("Model returned None result")
            
            if not isinstance(result, dict):
                raise ValueError(f"Model returned non-dict result: {type(result)}")
                
            if "messages" not in result:
                raise ValueError(f"Model result missing 'messages' key. Keys: {list(result.keys())}")
                
            if not result["messages"]:
                raise ValueError("Model result has empty messages list")

            # Success response
            response_content = result["messages"][-1].content
            response_payload = {
                "thread_id": config["configurable"]["thread_id"],  # Use the actual thread_id that worked
                "response": response_content
            }

            log_event("info", event="response_success", request_id=request_id, thread_id=thread_id)

            return {
                "statusCode": 200,
                "headers": CORS_HEADERS,
                "body": json.dumps(response_payload)
            }

    except Exception as e:
        # Log full stack trace once
        logger.exception("Unhandled error in lambda_handler")
        log_event("error", event="unhandled_exception", request_id=request_id, error=str(e))

        return {
            "statusCode": 500,
            "headers": CORS_HEADERS,
            "body": json.dumps({"error": str(e)})
        }