import json
import os
import psycopg2
import logging

logger = logging.getLogger()
handler = logging.StreamHandler()
formatter = logging.Formatter(
    "%(asctime)s - %(pathname)s - %(name)s - %(lineno)d - %(funcName)s- %(levelname)s  - %(message)s"
)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

def lambda_handler(event, context):
    """
    Lambda handler for the reset agent memory function
    """
    logger.info("Reset agent memory Lambda handler started")
    db_params = {
        "host": os.environ['DB_RDS'],
        "database": os.environ['DB_NAME'],
        "user": os.environ['DB_USER'],
        "password": os.environ['DB_PASS'],
        "port": int(os.environ.get('DB_PORT', 5432))
    }

    try:
        logger.info("Connecting to database")
        conn = psycopg2.connect(**db_params)
        cursor = conn.cursor()

        # TRUNCATE tables
        logger.info("Truncating tables")
        query = "TRUNCATE TABLE checkpoint_writes, checkpoint_blobs, checkpoints RESTART IDENTITY CASCADE;"
        cursor.execute(query)
        conn.commit()

        cursor.close()
        conn.close()

        logger.info("Tables truncated successfully")
        return {
            'statusCode': 200,
            'body': json.dumps({'message': 'Tables truncated successfully'})
        }

    except Exception as e:
        logger.error("Exception in reset agent memory lambda handler", exc_info=True)
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }