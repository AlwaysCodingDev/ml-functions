import boto3
import json

lambda_role_name = "JD-Generation-Stage-role-ag7794lc"
region = "eu-west-2"
account_id = "274267893716"
client_table_name = "Client-prwlo4vfg5d7xftuxlez5tukfe-NONE"


def attach_dynamodb_scan_policy():
    iam = boto3.client('iam', region_name=region)

    policy_name = 'LambdaDynamoDBScanPolicy'
    policy_document = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": ["dynamodb:Scan"],
                "Resource": f"arn:aws:dynamodb:{region}:{account_id}:table/{client_table_name}"
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
