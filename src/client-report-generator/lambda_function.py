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
import boto3
import json
# from botocore.exceptions import ClientError
# Configure logging
logger = logging.getLogger()
handler = logging.StreamHandler()
formatter = logging.Formatter(
    "%(asctime)s - %(pathname)s - %(name)s - %(lineno)d - %(funcName)s- %(levelname)s  - %(message)s"
)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# CORS Headers
CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "Content-Type, Authorization"
}




def lambda_handler(event, context):
    client_id = None

    try:
        logger.info("Client Report Generator Lambda function started")
        header = event.get("headers", {})
        auth_header = header.get("Authorization", "")  # Get the Authorization header

        if not auth_header:
            logger.warning("Authorization header missing")
            raise ValueError("Authorization header missing")

        
        parts = auth_header.split()
        if len(parts) != 2 or parts[0].lower() != "bearer":
            logger.warning("Invalid Authorization header format")
            raise ValueError("Invalid Authorization header format")

        token = parts[1]

        # Decode JWT
        decoded = jwt.decode(token, os.getenv("JWT_Secret"), algorithms=["HS256"])
        client_id = decoded.get("clientId", None)

        if not client_id:
            logger.warning("Client ID not found in token")
            raise ValueError("Client ID not found in token")

    except jwt.ExpiredSignatureError:
        logger.critical("Token has expired", exc_info=True)
        return {
            "statusCode": 401,
            "headers": CORS_HEADERS,
            "body": json.dumps({"error": "Token has expired"}),
        }

    except jwt.InvalidTokenError:
        logger.critical("Invalid token",exc_info=True)
        return {
            "statusCode": 401,
            "headers": CORS_HEADERS,
            "body": json.dumps({"error": "Invalid token"}),
        }

    except Exception as e:
        logger.critical("Exception in client report generator lambda function", exc_info=True)
        return {
            "statusCode": 401,
            "headers": CORS_HEADERS,
            "body": json.dumps({"error token": str(e)}),
        }
    try:

        table_names = {
            'clients': os.environ.get('CLIENTS_TABLE_NAME', 'Clients'),
            'jobs': os.environ.get('JOBS_TABLE_NAME', 'Jobs'),
            'responses': os.environ.get('RESPONSES_TABLE_NAME', 'Responses'),
            'vendors': os.environ.get('VENDORS_TABLE_NAME', 'Vendors'),
            'job_vendors': os.environ.get('JOB_VENDORS_TABLE_NAME', 'JobVendors')
        }
        output_bucket = os.environ.get('OUTPUT_BUCKET', 'your-output-bucket')
        logger.info("Fetching all data from DynamoDB...")

        raw_data = fetch_all_data(table_names)
        all_df = transform_data(raw_data)
        logger.info("Transforming data to required schema...")
        filtered = _filter_dataframes_for_client(all_df, client_id)
        logger.info("Filtering data for client...")
        excel_buffer = create_excel_file(filtered)
        file_name = f"client_{client_id}_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        upload_to_s3(excel_buffer, output_bucket, file_name)

        file_bytes = excel_buffer.getvalue()
        file_size_mb = len(file_bytes) / (1024 * 1024)
        is_api_gateway = 'requestContext' in event or 'httpMethod' in event
        size_limit_mb = 9.5 if is_api_gateway else 5.5

        headers = {
            'Content-Type': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            'Content-Disposition': f'attachment; filename="{file_name}"',
            'Content-Length': str(len(file_bytes)),
            'X-S3-Location': f's3://{output_bucket}/{file_name}',
            'X-S3-Download-URL': f'https://{output_bucket}.s3.amazonaws.com/{file_name}',
            'Access-Control-Allow-Origin': '*'
        }

        if file_size_mb > size_limit_mb:
            return _resp(200, {
                'message': 'File too large for inline return. Stored in S3.',
                'download_url': headers['X-S3-Download-URL'],
                's3_path': headers['X-S3-Location'],
                'size_mb': round(file_size_mb, 2)
            }, headers)

        return {
            'statusCode': 200,
            'headers': headers,
            'body': base64.b64encode(file_bytes).decode('utf-8'),
            'isBase64Encoded': True
        }

    except Exception as e:
        logger.error("Client export failed", exc_info=True)
        return _resp(500, {'error': str(e)})

# ==================== SHARED HELPERS ====================

def fetch_all_data(table_names: Dict[str, str]) -> Dict[str, List[Dict]]:
    """
    Fetch all data from DynamoDB tables
    """
    try:
        logger.info("Fetching all data from DynamoDB...")
        dynamodb = boto3.resource('dynamodb')
        data = {}
        for key, table_name in table_names.items():
            table = dynamodb.Table(table_name)
            items = []
            scan_kwargs = {}
            while True:
                response = table.scan(**scan_kwargs)
                items.extend(response.get('Items', []))
                if 'LastEvaluatedKey' not in response:
                    break
                scan_kwargs['ExclusiveStartKey'] = response['LastEvaluatedKey']
            data[key] = items
        return data
    except Exception as e:
        logger.error("Failed to fetch data from DynamoDB", exc_info=True)
        raise

def transform_data(raw_data: Dict[str, List[Dict]]) -> Dict[str, pd.DataFrame]:
    """
    Transform raw data to required schema
    """
    try:
        logger.info("Transforming data to required schema...")
        transformed = {}
        clients_data, clients_job_ids_data = transform_clients(raw_data.get('clients', []))
        transformed['Clients'] = pd.DataFrame(clients_data)
        transformed['Clients_JobIds'] = pd.DataFrame(clients_job_ids_data)
        transformed['JobPrompts'] = pd.DataFrame(transform_jobs(raw_data.get('jobs', [])))
        transformed['JobResponses'] = pd.DataFrame(transform_responses(raw_data.get('responses', [])))
        vendors_data, vendor_tags_data = transform_vendors(raw_data.get('vendors', []))
        transformed['Vendors'] = pd.DataFrame(vendors_data)
        transformed['Vendor_Tags'] = pd.DataFrame(vendor_tags_data)
        transformed['JobVendor'] = pd.DataFrame(transform_job_vendors(raw_data.get('job_vendors', [])))
        return transformed
    except Exception as e:
        logger.error("Failed to transform data", exc_info=True)
        raise

def transform_clients(clients: List[Dict]) -> tuple[List[Dict], List[Dict]]:
    """
    Transform clients data to required schema
    """
    try:
        logger.info("Transforming clients data to required schema...")
        clients_data = []
        clients_job_ids_data = []
        for client in clients:
            clients_data.append({
                'client_id': client.get('id', ''),
                'company_name': client.get('companyName', ''),
                'email': client.get('email', ''),
                'phone_number': client.get('phoneNumber', ''),
                'is_verified': client.get('isVerified', False),
                'created_at': parse_datetime(client.get('createdAt')),
                'updated_at': parse_datetime(client.get('updatedAt'))
            })
            job_ids = client.get('jobIds', [])
            if isinstance(job_ids, str):
                try:
                    job_ids = json.loads(job_ids)
                except:
                    job_ids = []
            if isinstance(job_ids, list):
                for job_id in job_ids:
                    clients_job_ids_data.append({'client_id': client.get('id', ''), 'job_id': job_id})
        return clients_data, clients_job_ids_data
    except Exception as e:
        logger.error("Failed to transform clients data", exc_info=True)
        raise

def transform_jobs(jobs: List[Dict]) -> List[Dict]:
    """
    Transform jobs data to required schema
    """
    try:
        logger.info("Transforming jobs data to required schema...")
        out = []
        for job in jobs:
            out.append({
                'job_id': job.get('id', ''),
                'client_id': job.get('clientId', ''),
                'company_name': job.get('companyName', ''),
                'job_role': job.get('jobRole', ''),
                'job_location': job.get('jobLocation', ''),
                'employment_type': job.get('employmentType', ''),
                'education_requirements': job.get('educationRequirements', ''),
                'salary_range': job.get('salaryRange', ''),
                'years_of_experience': safe_int(job.get('yearsOfExperience')),
                'vacancy': safe_int(job.get('vacancy')),
                'job_deadline': parse_datetime(job.get('jobDeadline')),
                'video_interview_deadline': safe_int(job.get('videoInterviewDeadline')),
                'audio_interview_deadline': safe_int(job.get('audioInterviewDeadline')),
                'is_open': job.get('isOpen', True),
                'hiring_complete': job.get('hiringComplete', False),
                'created_at': parse_datetime(job.get('createdAt')),
                'updated_at': parse_datetime(job.get('updatedAt'))
            })
        return out
    except Exception as e:
        logger.error("Failed to transform jobs data", exc_info=True)
        raise

def transform_responses(responses: List[Dict]) -> List[Dict]:
    """
    Transform responses data to required schema
    """
    try:
        logger.info("Transforming responses data to required schema...")
        out = []
        for r in responses:
            out.append({
                'response_id': r.get('id', ''),
                'candidate_id': r.get('candidateId', ''),
                'job_id': r.get('jobId', ''),
                'client_id': r.get('clientId', ''),
                'vendor_id': r.get('vendorId', ''),
                'created_at': parse_datetime(r.get('createdAt')),
                'updated_at': parse_datetime(r.get('updatedAt')),
                'audio_score': safe_float(r.get('audioScore')),
                'video_score': safe_float(r.get('videoScore')),
                'body_lang_score': safe_float(r.get('bodyLangScore')),
                'similarity_score': safe_float(r.get('similarityScore')),
                'final_selection': r.get('finalSelection', False),
                'resume_submitted': bool(r.get('resume')),
                'interview_completed': r.get('videoInterviewCompleted', False) or r.get('audioInterviewCompleted', False),
                'evaluation_completed': r.get('evaluationCompletion') == 'completed',
                'report_completed': r.get('reportCompletion') == 'completed'
            })
        return out
    except Exception as e:
        logger.error("Failed to transform responses data", exc_info=True)
        raise

def transform_vendors(vendors: List[Dict]) -> tuple[List[Dict], List[Dict]]:
    """
    Transform vendors data to required schema
    """
    try:
        logger.info("Transforming vendors data to required schema...")
        vendors_data = []
        vendor_tags_data = []
        for v in vendors:
            vendors_data.append({
                'vendor_id': v.get('id', ''),
                'agency_name': v.get('agencyName', ''),
                'region': str(v.get('region', '')),
                'is_verified': v.get('isVerified', False),
                'created_at': parse_datetime(v.get('createdAt')),
                'updated_at': parse_datetime(v.get('updatedAt'))
            })
            tags = v.get('tags', '')
            if tags:
                if isinstance(tags, str):
                    try:
                        tags = json.loads(tags)
                    except:
                        tags = [t.strip() for t in tags.split(',')]
                if isinstance(tags, list):
                    for tag in tags:
                        vendor_tags_data.append({'vendor_id': v.get('id', ''), 'tag': str(tag)})
        return vendors_data, vendor_tags_data
    except Exception as e:
        logger.error("Failed to transform vendors data", exc_info=True)
        raise

def transform_job_vendors(job_vendors: List[Dict]) -> List[Dict]:
    """
    Transform job vendors data to required schema
    """
    try:
        logger.info("Transforming job vendors data to required schema...")
        return [{
            'job_id': jv.get('jobId', ''),
            'vendor_id': jv.get('vendorId', ''),
            'created_at': parse_datetime(jv.get('createdAt')),
            'updated_at': parse_datetime(jv.get('updatedAt'))
        } for jv in job_vendors]
    except Exception as e:
        logger.error("Failed to transform job vendors data", exc_info=True)
        raise

def parse_datetime(val: Any) -> Optional[str]:
    """
    Parse datetime value
    """
    try:
        if isinstance(val, str):
            return pd.to_datetime(val).strftime('%Y-%m-%d %H:%M:%S')
        elif isinstance(val, (int, float)):
            return datetime.fromtimestamp(val).strftime('%Y-%m-%d %H:%M:%S')
    except Exception:
        logger.error("Failed to parse datetime", exc_info=True)
        return None

def safe_int(val: Any) -> int:
    """
    Convert value to int
    """
    try:
        return int(val)
    except Exception:
        logger.error("Failed to convert to int", exc_info=True)
        return 0

def safe_float(val: Any) -> Optional[float]:
    """
    Convert value to float
    """
    try:
        return float(val)
    except Exception:
        logger.error("Failed to convert to float", exc_info=True)
        return None

def create_excel_file(dataframes: Dict[str, pd.DataFrame]) -> BytesIO:
    """
    Create Excel file from dataframes
    """
    try:
        logger.info("Creating Excel file...")
        excel_buffer = BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            for sheet_name, df in dataframes.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)
        excel_buffer.seek(0)
        return excel_buffer
    except Exception as e:
        logger.error("Failed to create Excel file", exc_info=True)
        raise

def upload_to_s3(buffer: BytesIO, bucket: str, key: str):
    """
    Upload file to S3
    """
    try:
        logger.info("Uploading file to S3...")
        boto3.client('s3').put_object(
            Bucket=bucket,
            Key=key,
            Body=buffer.getvalue(),
            ContentType='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
    except Exception as e:
        logger.error("Failed to upload file to S3", exc_info=True)
        raise

# TODO Not found any reference
def _extract_client_id(event: Dict[str, Any]) -> Optional[str]:
    """
    Extract client ID from event
    """
    try:
        q = event.get('queryStringParameters') or {}
        if isinstance(q, dict) and q.get('client_id'):
            return q['client_id']
        body = event.get('body')
        if body:
            try:
                return json.loads(body).get('client_id')
            except:
                return None
        return None
    except Exception as e:
        logger.error("Failed to extract client ID", exc_info=True)
        return None

def _filter_dataframes_for_client(df: Dict[str, pd.DataFrame], client_id: str) -> Dict[str, pd.DataFrame]:
    """
    Filter dataframes for client ID
    """
    try:
        jobs_f = df['JobPrompts'][df['JobPrompts']['client_id'] == client_id]
        job_ids = jobs_f['job_id'].tolist()
        responses_f = df['JobResponses'][df['JobResponses']['client_id'] == client_id]
        cji_f = df['Clients_JobIds'][df['Clients_JobIds']['client_id'] == client_id]
        clients_f = df['Clients'][df['Clients']['client_id'] == client_id]
        jv_f = df['JobVendor'][df['JobVendor']['job_id'].isin(job_ids)]
        vendors_f = df['Vendors'][df['Vendors']['vendor_id'].isin(jv_f['vendor_id'])]
        vtags_f = df['Vendor_Tags'][df['Vendor_Tags']['vendor_id'].isin(vendors_f['vendor_id'])]
        return {
            'Clients': clients_f.reset_index(drop=True),
            'Clients_JobIds': cji_f.reset_index(drop=True),
            'JobPrompts': jobs_f.reset_index(drop=True),
            'JobResponses': responses_f.reset_index(drop=True),
            'JobVendor': jv_f.reset_index(drop=True),
            'Vendors': vendors_f.reset_index(drop=True),
            'Vendor_Tags': vtags_f.reset_index(drop=True),
        }
    except Exception as e:
        logger.error("Failed to filter dataframes for client ID", exc_info=True)
        raise

def _resp(status: int, body: dict, headers: Optional[dict] = None):
    """
    Return response
    """
    try:
        h = {'Content-Type': 'application/json'}
        if headers:
            h.update(headers)
        return {'statusCode': status, 'headers': h, 'body': json.dumps(body)}
    except Exception as e:
        logger.error("Failed to return response", exc_info=True)
        raise