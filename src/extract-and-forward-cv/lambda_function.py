import os
import boto3
import fitz 
from botocore.exceptions import BotoCoreError, ClientError
import json
import time

s3 = boto3.client("s3")
dynamodb = boto3.resource("dynamodb")
lambda_client = boto3.client("lambda")
logs_client = boto3.client("logs")

DDB_TABLE_NAME = os.environ["DDB"]
DDB_TABLE_NAME2 = os.environ["DDB2"]

# --------------------------
# Logging utilities
# --------------------------
def put_log(group, stream, message):
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


# --------------------------
# Lambda Handler
# --------------------------
def lambda_handler(event, context):
    """
    Lambda handler for extracting resume text from PDF and updating DynamoDB.
    
    """

    log_to_destinations(None, None, f"START lambda_handler with event={event}")

    table = dynamodb.Table(DDB_TABLE_NAME)

    for record in event.get("Records", []):
        
        if record.get("eventName") != "INSERT":
            continue

        new_image = record["dynamodb"].get("NewImage", {})
        if not all(k in new_image for k in ("id", "bucket", "fileName", "jobId")):
            msg = f"Skipping record—missing required attributes: {new_image}"
            print(msg)
            log_to_destinations(None, None, msg)
            continue

        response_id = new_image["id"]["S"]
        bucket      = new_image["bucket"]["S"]
        key         = new_image["fileName"]["S"]
        job_id      = new_image["jobId"]["S"]

        log_to_destinations(job_id, response_id, f"Processing response_id={response_id}, job_id={job_id}, key={key}")

        if "resumeText" in new_image and new_image["resumeText"].get("S"):
            msg = f"response_id={response_id}: resume_text already exists; skipping extraction."
            print(msg)
            log_to_destinations(job_id, response_id, msg)
            continue

        try:
            s3_obj = s3.get_object(Bucket=bucket, Key=key)
            pdf_data = s3_obj["Body"].read()

            doc = fitz.open(stream=pdf_data, filetype="pdf")
            extracted_text = "".join(page.get_text() for page in doc)

            table.update_item(
                Key={"id": response_id},
                UpdateExpression="SET resumeText = :r",
                ExpressionAttributeValues={":r": extracted_text},
            )

            msg = f"id={response_id}: resume_text updated successfully."
            print(msg)
            log_to_destinations(job_id, response_id, msg)

        except (ClientError, BotoCoreError) as aws_err:
            msg = f"[AWS Error] response_id={response_id}: {aws_err}"
            print(msg)
            log_to_destinations(job_id, response_id, msg)

        except Exception as e:
            msg = f"[Error] response_id={response_id}: Failed to extract/update resume_text: {e}"
            print(msg)
            log_to_destinations(job_id, response_id, msg)

        try:
            response = lambda_client.invoke(
                FunctionName="arn:aws:lambda:eu-west-2:274267893716:function:EmbeddingFunction",
                InvocationType="Event",  # async
                Payload=json.dumps({
                    "response_id": response_id,
                    "job_id": job_id,
                    "job_table": DDB_TABLE_NAME2,
                    "response_table": DDB_TABLE_NAME
                })
            )
            log_to_destinations(job_id, response_id, f"Invoked EmbeddingFunction for response_id={response_id}")

        except Exception as e:
            msg = f"[Error] response_id={response_id}: Failed to invoke target Lambda: {e}"
            print(msg)
            log_to_destinations(job_id, response_id, msg)

    log_to_destinations(None, None, "END lambda_handler")
    return {"statusCode": 200, "body": "Processed DynamoDB stream successfully."}