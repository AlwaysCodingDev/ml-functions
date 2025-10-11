import json
import boto3
import pandas as pd
from io import BytesIO
import logging
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
import base64
import jwt

# Configure logging
logger = logging.getLogger()
handler = logging.StreamHandler()
formatter = logging.Formatter(
    "%(asctime)s - %(pathname)s - %(name)s - %(lineno)d - %(funcName)s- %(levelname)s  - %(message)s"
)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Example usage
# lambda_role_name = "Admin-Report-Stage-role-dwvwmovp"
# region = "eu-west-2"
# account_id = "274267893716"
# table_name = "Admin-prwlo4vfg5d7xftuxlez5tukfe-NONE"




# def attach_dynamodb_scan_policy(lambda_role_name, region, account_id, table_name):
#     iam = boto3.client('iam')

#     policy_name = 'LambdaDynamoDBScanPolicy'

#     # Define the policy document allowing dynamodb:Scan on the specific table ARN
#     policy_document = {
#         "Version": "2012-10-17",
#         "Statement": [
#             {
#                 "Effect": "Allow",
#                 "Action": [
#                     "dynamodb:Scan"
#                 ],
#                 "Resource": f"arn:aws:dynamodb:{region}:{account_id}:table/{table_name}"
#             }
#         ]
#     }

#     # Put (create or update) the inline policy attached to the role
#     response = iam.update_assume_role_policy(
#         RoleName=lambda_role_name,
#         # PolicyName=policy_name,
#         PolicyDocument=json.dumps(policy_document)
#     )

#     print("Policy attached successfully:", response)


# attach_dynamodb_scan_policy(lambda_role_name, region, account_id, table_name)

# def get_secret(key= "", jwt=False):
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

#     return secret

# os.environ["JWT_Secret"] = get_secret(jwt=True)

def lambda_handler(event, context):
    """
    Lambda function that ALWAYS returns Excel file directly in response
    Requires valid JWT token with admin verification
    """
    try:
        logger.info("Admin Report Generator Lambda function started")
        # Verify JWT token first
        admin_verification = verify_admin_token(event)
        if not admin_verification["valid"]:
            logger.warning("Admin verification failed: %s", admin_verification["message"])
            
            error_response = {
                "error": "Unauthorized",
                "message": admin_verification["message"],
            }

            # Detect if request is from API Gateway for proper response format
            is_api_gateway = "requestContext" in event or "httpMethod" in event

            if is_api_gateway:
                return {
                    "statusCode": 401,
                    "headers": {
                        "Content-Type": "application/json",
                        "Access-Control-Allow-Origin": "*",
                        "Access-Control-Allow-Headers": "Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token",
                        "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
                    },
                    "body": json.dumps(error_response),
                }
            else:
                return {
                    "statusCode": 401,
                    "headers": {"Content-Type": "application/json"},
                    "body": json.dumps(error_response),
                }

        logger.info("Admin verified: %s", admin_verification["admin_id"])

        # Get table names from environment variables
        table_names = {
            "clients": os.environ.get("CLIENTS_TABLE_NAME", "Clients"),
            "jobs": os.environ.get("JOBS_TABLE_NAME", "Jobs"),
            "responses": os.environ.get("RESPONSES_TABLE_NAME", "Responses"),
            "vendors": os.environ.get("VENDORS_TABLE_NAME", "Vendors"),
            "job_vendors": os.environ.get("JOB_VENDORS_TABLE_NAME", "JobVendors"),
        }

        output_bucket = os.environ.get("OUTPUT_BUCKET", "your-output-bucket")

        # Fetch all data from DynamoDB
        logger.info("Starting DynamoDB data fetch...")
        raw_data = fetch_all_data(table_names)

        # Transform data to required schema
        logger.info("Transforming data to required schema...")
        transformed_data = transform_data(raw_data)

        logger.info("Creating Excel file...")
        excel_buffer = create_excel_file(transformed_data)

        # ALWAYS upload to S3 first (backup/archive)
        file_name = f"database_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        upload_to_s3(excel_buffer, output_bucket, file_name)
        logger.info(f"File uploaded to S3: s3://{output_bucket}/{file_name}")

        # Detect if request is from API Gateway
        is_api_gateway = "requestContext" in event or "httpMethod" in event

        # Get file content and size
        file_content = excel_buffer.getvalue()
        file_size_mb = len(file_content) / (1024 * 1024)

        logger.info(f"Excel file created, size: {file_size_mb:.2f}MB")

        # Set size limits based on invocation method
        size_limit_mb = 9.5 if is_api_gateway else 5.5  # API Gateway: 10MB, Lambda: 6MB
        limit_name = "API Gateway" if is_api_gateway else "Lambda"

        # Check if file is too large for direct return
        if file_size_mb > size_limit_mb:
            error_response = {
                "message": f"File too large for {limit_name} direct return but stored in S3",
                "file_location": f"s3://{output_bucket}/{file_name}",
                "download_url": f"https://{output_bucket}.s3.amazonaws.com/{file_name}",
                "file_size_mb": round(file_size_mb, 2),
                "max_direct_size_mb": size_limit_mb,
                "records_count": {k: len(v) for k, v in transformed_data.items()},
                "stored_in_s3": True,
                "admin_id": admin_verification["admin_id"],
            }

            if is_api_gateway:
                return {
                    "statusCode": 200,  # Changed from 413 to 200 since file is available in S3
                    "headers": {
                        "Content-Type": "application/json",
                        "Access-Control-Allow-Origin": "*",
                        "Access-Control-Allow-Headers": "Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token",
                        "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
                    },
                    "body": json.dumps(error_response),
                }
            else:
                return {
                    "statusCode": 200,  # Changed from 413 to 200 since file is available in S3
                    "headers": {"Content-Type": "application/json"},
                    "body": json.dumps(error_response),
                }

        # Return file as base64 encoded response (also stored in S3)
        file_base64 = base64.b64encode(file_content).decode("utf-8")

        if is_api_gateway:
            # API Gateway compatible response with S3 info in headers
            return {
                "statusCode": 200,
                "headers": {
                    "Content-Type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    "Content-Disposition": f'attachment; filename="{file_name}"',
                    "Content-Length": str(len(file_content)),
                    "X-S3-Location": f"s3://{output_bucket}/{file_name}",
                    "X-S3-Download-URL": f"https://{output_bucket}.s3.amazonaws.com/{file_name}",
                    "X-Admin-ID": admin_verification["admin_id"],
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Headers": "Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token",
                    "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
                    "Access-Control-Expose-Headers": "Content-Disposition,Content-Length,X-S3-Location,X-S3-Download-URL,X-Admin-ID",
                },
                "body": file_base64,
                "isBase64Encoded": True,
            }
        else:
            # Direct Lambda invocation response with S3 info in headers
            return {
                "statusCode": 200,
                "headers": {
                    "Content-Type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    "Content-Disposition": f'attachment; filename="{file_name}"',
                    "Content-Length": str(len(file_content)),
                    "X-S3-Location": f"s3://{output_bucket}/{file_name}",
                    "X-S3-Download-URL": f"https://{output_bucket}.s3.amazonaws.com/{file_name}",
                    "X-Admin-ID": admin_verification["admin_id"],
                },
                "body": file_base64,
                "isBase64Encoded": True,
            }

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": str(e)}),
        }


def verify_admin_token(event) -> Dict[str, Any]:
    """
    Verify JWT token and check if admin exists in DynamoDB
    Returns dict with validation result and admin info
    """
    try:
        # Extract token from event
        token = extract_token_from_event(event)
        if not token:
            return {"valid": False, "message": "No authorization token provided"}

        # Get JWT secret from environment
        jwt_secret = os.environ.get("JWT_Secret")
        if not jwt_secret:
            logger.error("JWT_SECRET environment variable not set")
            return {"valid": False, "message": "Server configuration error"}

        # Get admin table name from environment
        admin_table_name = os.environ.get("ADMIN_TABLE_NAME", "Admins")

        # Decode JWT token
        try:
            decoded_token = jwt.decode(token, jwt_secret, algorithms=["HS256"])
            admin_id = (
                decoded_token.get("adminId")
                or decoded_token.get("id")
                or decoded_token.get("sub")
            )

            if not admin_id:
                return {"valid": False, "message": "Invalid token: missing admin ID"}

        except jwt.ExpiredSignatureError:
            return {"valid": False, "message": "Token has expired"}
        except jwt.InvalidTokenError as e:
            logger.error(f"JWT decode error: {str(e)}")
            return {"valid": False, "message": "Invalid token"}

        # Check if admin exists in DynamoDB
        dynamodb = boto3.resource("dynamodb")
        admin_table = dynamodb.Table(admin_table_name)

        try:
            response = admin_table.get_item(Key={"id": admin_id})

            if "Item" not in response:
                logger.warning(f"Admin ID {admin_id} not found in {admin_table_name}")
                return {"valid": False, "message": "Admin not found"}

            admin_item = response["Item"]

            # Optional: Check if admin is active/enabled
            if (
                admin_item.get("isActive") is False
                or admin_item.get("status") == "disabled"
            ):
                return {"valid": False, "message": "Admin account is disabled"}

            logger.info(f"Admin verified successfully: {admin_id}")
            return {"valid": True, "admin_id": admin_id, "admin_data": admin_item}

        except Exception as e:
            logger.error(f"DynamoDB error while checking admin: {str(e)}")
            return {
                "valid": False,
                "message": "Database error during admin verification",
            }

    except Exception as e:
        logger.error(f"Error in admin verification: {str(e)}")
        return {"valid": False, "message": "Authentication error"}


def extract_token_from_event(event) -> Optional[str]:
    """
    Extract JWT token from various event sources
    """
    # API Gateway - Authorization header
    if "headers" in event:
        headers = event["headers"]

        # Check Authorization header (case-insensitive)
        for header_name, header_value in headers.items():
            if header_name.lower() == "authorization":
                if header_value.startswith("Bearer "):
                    return header_value[7:]  # Remove 'Bearer ' prefix
                else:
                    return header_value

        # Check x-access-token header
        for header_name, header_value in headers.items():
            if header_name.lower() == "x-access-token":
                return header_value

    # Query parameters
    if "queryStringParameters" in event and event["queryStringParameters"]:
        token = event["queryStringParameters"].get("token")
        if token:
            return token

    # Request body (for POST requests)
    if "body" in event and event["body"]:
        try:
            if isinstance(event["body"], str):
                body = json.loads(event["body"])
            else:
                body = event["body"]

            if "token" in body:
                return body["token"]
        except:
            pass

    # Direct Lambda invocation - check event directly
    if "token" in event:
        return event["token"]

    return None


def fetch_all_data(table_names: Dict[str, str]) -> Dict[str, List[Dict]]:
    """
    Fetch all data from DynamoDB tables
    """
    try:
        logger.info("Fetching data from DynamoDB tables...")
        dynamodb = boto3.resource("dynamodb")
        data = {}

        for key, table_name in table_names.items():
            logger.info("Fetching data from %s...", table_name)
            table = dynamodb.Table(table_name)

            items = []
            scan_kwargs = {}

            # Scan with pagination
            while True:
                response = table.scan(**scan_kwargs)
                items.extend(response.get("Items", []))

                if "LastEvaluatedKey" not in response:
                    break
                scan_kwargs["ExclusiveStartKey"] = response["LastEvaluatedKey"]

            data[key] = items
            logger.info("Fetched %s records from %s", len(items), table_name)
        return data
    except Exception as e:
        logger.critical("Failed to fetch data from DynamoDB tables:", exc_info=True)
        raise



def transform_data(raw_data: Dict[str, List[Dict]]) -> Dict[str, pd.DataFrame]:
    """
    Transform raw DynamoDB data to required schema format
    """
    try:
        logger.info("Transforming data...")
        transformed = {}

        # Transform Clients data
        clients_data, clients_job_ids_data = transform_clients(raw_data.get("clients", []))
        transformed["Clients"] = pd.DataFrame(clients_data)
        transformed["Clients_JobIds"] = pd.DataFrame(clients_job_ids_data)

        # Transform Jobs data
        transformed["JobPrompts"] = pd.DataFrame(transform_jobs(raw_data.get("jobs", [])))

        # Transform Responses data
        transformed["JobResponses"] = pd.DataFrame(
            transform_responses(raw_data.get("responses", []))
        )

        # Transform Vendors data
        vendors_data, vendor_tags_data = transform_vendors(raw_data.get("vendors", []))
        transformed["Vendors"] = pd.DataFrame(vendors_data)
        transformed["Vendor_Tags"] = pd.DataFrame(vendor_tags_data)

        # Transform JobVendors data
        transformed["JobVendor"] = pd.DataFrame(
            transform_job_vendors(raw_data.get("job_vendors", []))
        )

        return transformed
    except Exception as e:
        logger.critical("Failed to transform data:", exc_info=True)
        raise

def transform_clients(clients: List[Dict]) -> tuple[List[Dict], List[Dict]]:
    """Transform clients data"""
    try:
        logger.info("Transforming clients data...")
        clients_data = []
        clients_job_ids_data = []

        for client in clients:
            # Main client record
            client_record = {
                "client_id": client.get("id", ""),
                "company_name": client.get("companyName", ""),
                "email": client.get("email", ""),
                "phone_number": client.get("phoneNumber", ""),
                "is_verified": client.get("isVerified", False),
                "created_at": parse_datetime(client.get("createdAt")),
                "updated_at": parse_datetime(client.get("updatedAt")),
            }
            clients_data.append(client_record)

            # Extract job IDs
            job_ids = client.get("jobIds", [])
            if isinstance(job_ids, str):
                try:
                    job_ids = json.loads(job_ids)
                except:
                    job_ids = []

            if isinstance(job_ids, list):
                for job_id in job_ids:
                    clients_job_ids_data.append(
                        {"client_id": client.get("id", ""), "job_id": job_id}
                    )

        return clients_data, clients_job_ids_data
    except Exception as e:
        logger.critical("Failed to transform clients data:", exc_info=True)
        raise


def transform_jobs(jobs: List[Dict]) -> List[Dict]:
    """Transform jobs data"""
    try:
        logger.info("Transforming jobs data...")
        job_prompts_data = []   

        for job in jobs:
            job_record = {
                "job_id": job.get("id", ""),
                "client_id": job.get("clientId", ""),
                "company_name": job.get("companyName", ""),
                "job_role": job.get("jobRole", ""),
                "job_location": job.get("jobLocation", ""),
                "employment_type": job.get("employmentType", ""),
                "education_requirements": job.get("educationRequirements", ""),
                "salary_range": job.get("salaryRange", ""),
                "years_of_experience": safe_int(job.get("yearsOfExperience", 0)),
                "vacancy": safe_int(job.get("vacancy", 1)),
                "job_deadline": parse_datetime(job.get("jobDeadline")),
                "video_interview_deadline": safe_int(job.get("videoInterviewDeadline")),
                "audio_interview_deadline": safe_int(job.get("audioInterviewDeadline")),
                "is_open": job.get("isOpen", True),
                "hiring_complete": job.get("hiringComplete", False),
                "created_at": parse_datetime(job.get("createdAt")),
                "updated_at": parse_datetime(job.get("updatedAt")),
            }
            job_prompts_data.append(job_record)

        return job_prompts_data
    except Exception as e:
        logger.critical("Failed to transform jobs data:", exc_info=True)
        raise

def transform_responses(responses: List[Dict]) -> List[Dict]:
    """Transform responses data"""
    try:
        logger.info("Transforming responses data...")
        job_responses_data = []

        for response in responses:
            response_record = {
                "response_id": response.get("id", ""),
                "candidate_id": response.get("candidateId", ""),
                "job_id": response.get("jobId", ""),
                "client_id": response.get("clientId", ""),
                "vendor_id": response.get("vendorId", ""),
                "created_at": parse_datetime(response.get("createdAt")),
                "updated_at": parse_datetime(response.get("updatedAt")),
                "audio_score": safe_float(response.get("audioScore")),
                "video_score": safe_float(response.get("videoScore")),
                "body_lang_score": safe_float(response.get("bodyLangScore")),
                "similarity_score": safe_float(response.get("similarityScore")),
                "final_selection": response.get("finalSelection", False),
                "resume_submitted": bool(response.get("resume", "")),
                "interview_completed": response.get("videoInterviewCompleted", False)
                or response.get("audioInterviewCompleted", False),
                "evaluation_completed": response.get("evaluationCompletion") == "completed",
                "report_completed": response.get("reportCompletion") == "completed",
            }
            job_responses_data.append(response_record)

        return job_responses_data
    except Exception as e:
        logger.critical("Failed to transform responses data:", exc_info=True)
        raise


def transform_vendors(vendors: List[Dict]) -> tuple[List[Dict], List[Dict]]:
    """Transform vendors data"""
    try:
        logger.info("Transforming vendors data...")
        vendors_data = []
        vendor_tags_data = []

        for vendor in vendors:
            # Main vendor record
            vendor_record = {
                "vendor_id": vendor.get("id", ""),
                "agency_name": vendor.get("agencyName", ""),
                "region": str(vendor.get("region", "")),
                "is_verified": vendor.get("isVerified", False),
                "created_at": parse_datetime(vendor.get("createdAt")),
                "updated_at": parse_datetime(vendor.get("updatedAt")),
            }
            vendors_data.append(vendor_record)

            # Extract tags
            tags = vendor.get("tags", "")
            if tags:
                if isinstance(tags, str):
                    try:
                        tags = json.loads(tags)
                    except:
                        tags = [tag.strip() for tag in tags.split(",") if tag.strip()]

                if isinstance(tags, list):
                    for tag in tags:
                        vendor_tags_data.append(
                            {"vendor_id": vendor.get("id", ""), "tag": str(tag)}
                        )

        return vendors_data, vendor_tags_data
    except Exception as e:
        logger.critical("Failed to transform vendors data:", exc_info=True)
        raise


def transform_job_vendors(job_vendors: List[Dict]) -> List[Dict]:
    """Transform job vendors data"""
    try:
        logger.info("Transforming job vendors data...")
        job_vendor_data = []

        for jv in job_vendors:
            job_vendor_record = {
                "job_id": jv.get("jobId", ""),
                "vendor_id": jv.get("vendorId", ""),
                "created_at": parse_datetime(jv.get("createdAt")),
                "updated_at": parse_datetime(jv.get("updatedAt")),
            }
            job_vendor_data.append(job_vendor_record)

        return job_vendor_data
    except Exception as e:
        logger.critical("Failed to transform job vendors data:", exc_info=True)
        raise

def parse_datetime(datetime_str: Any) -> Optional[str]:
    """Parse datetime to string format"""
    if not datetime_str:
        return None

    try:
        if isinstance(datetime_str, str):
            return pd.to_datetime(datetime_str).strftime("%Y-%m-%d %H:%M:%S")
        elif isinstance(datetime_str, (int, float)):
            return datetime.fromtimestamp(datetime_str).strftime("%Y-%m-%d %H:%M:%S")
    except:
        logger.error("Failed to parse datetime %s",datetime_str)
        return None


def safe_int(value: Any) -> int:
    """Safely convert to int"""
    try:
        return int(value) if value is not None else 0
    except (ValueError, TypeError):
        logger.error("Failed to convert %s to int",value)
        return 0


def safe_float(value: Any) -> Optional[float]:
    """Safely convert to float"""
    try:
        return float(value) if value is not None else None
    except (ValueError, TypeError):
        logger.error("Failed to convert %s to float",value)
        return None


def create_excel_file(dataframes: Dict[str, pd.DataFrame]) -> BytesIO:
    """Create Excel file with multiple sheets"""
    try:
        logger.info("Creating Excel file...")
        excel_buffer = BytesIO()

        with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
            for sheet_name, df in dataframes.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)

        excel_buffer.seek(0)
        return excel_buffer
    except Exception as e:
        logger.critical("Failed to create Excel file:", exc_info=True)
        raise


def upload_to_s3(excel_buffer: BytesIO, bucket: str, file_name: str):
    """Upload Excel file to S3"""
    try:
        logger.info("Uploading Excel file to S3...")
        s3_client = boto3.client("s3")

        s3_client.put_object(
            Bucket=bucket,
            Key=file_name,
            Body=excel_buffer.getvalue(),
        ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
        logger.info(f"File uploaded to s3://{bucket}/{file_name}")
    except Exception as e:
        logger.critical("Failed to upload Excel file to S3:", exc_info=True)
        raise