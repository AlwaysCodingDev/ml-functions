import boto3
import json

lambda_role_name = "Admin-Report-Stage-role-dwvwmovp"
region = "eu-west-2"
account_id = "274267893716"
admin_table_name = "Admin-prwlo4vfg5d7xftuxlez5tukfe-NONE"
client_table_name = "Client-prwlo4vfg5d7xftuxlez5tukfe-NONE"
job_table="JobPrompt-prwlo4vfg5d7xftuxlez5tukfe-NONE"
JOB_VENDORS_TABLE_NAME="JobVendor-prwlo4vfg5d7xftuxlez5tukfe-NONE"
RESPONSES_TABLE_NAME="JobResponse-prwlo4vfg5d7xftuxlez5tukfe-NONE"
VENDORS_TABLE_NAME="Vendor-prwlo4vfg5d7xftuxlez5tukfe-NONE"

def attach_dynamodb_scan_policy():
    iam = boto3.client('iam', region_name=region)

    policy_name = 'LambdaDynamoDBScanPolicy'
    policy_document = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": ["dynamodb:Scan"],
                "Resource": f"arn:aws:dynamodb:{region}:{account_id}:table/{admin_table_name}"
            },
             {
                "Effect": "Allow",
                "Action": ["dynamodb:Scan"],
                "Resource": f"arn:aws:dynamodb:{region}:{account_id}:table/{client_table_name}"
            },
            {
                "Effect": "Allow",
                "Action": ["dynamodb:Scan"],
                "Resource": f"arn:aws:dynamodb:{region}:{account_id}:table/{job_table}"
            },
             {
                "Effect": "Allow",
                "Action": ["dynamodb:Scan"],
                "Resource": f"arn:aws:dynamodb:{region}:{account_id}:table/{JOB_VENDORS_TABLE_NAME}"
            },
             {
                "Effect": "Allow",
                "Action": ["dynamodb:Scan"],
                "Resource": f"arn:aws:dynamodb:{region}:{account_id}:table/{RESPONSES_TABLE_NAME}"
            },
             {
                "Effect": "Allow",
                "Action": ["dynamodb:Scan"],
                "Resource": f"arn:aws:dynamodb:{region}:{account_id}:table/{VENDORS_TABLE_NAME}"
            }
        ]
    }

    response = iam.put_role_policy(
        RoleName=lambda_role_name,
        PolicyName=policy_name,
        PolicyDocument=json.dumps(policy_document)
    )

    print("Policy attached successfully:", response)

attach_dynamodb_scan_policy()
