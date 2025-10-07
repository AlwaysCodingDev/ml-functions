import json
import os
import psycopg2

def lambda_handler(event, context):
    
    db_params = {
        "host": os.environ['DB_RDS'],
        "database": os.environ['DB_NAME'],
        "user": os.environ['DB_USER'],
        "password": os.environ['DB_PASS'],
        "port": int(os.environ.get('DB_PORT', 5432))
    }

    try:
        conn = psycopg2.connect(**db_params)
        cursor = conn.cursor()

        # TRUNCATE tables
        query = "TRUNCATE TABLE checkpoint_writes, checkpoint_blobs, checkpoints RESTART IDENTITY CASCADE;"
        cursor.execute(query)
        conn.commit()

        cursor.close()
        conn.close()

        return {
            'statusCode': 200,
            'body': json.dumps({'message': 'Tables truncated successfully'})
        }

    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }