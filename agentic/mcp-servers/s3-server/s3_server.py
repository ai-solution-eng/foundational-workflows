import os
import json
import base64
import logging
from typing import Any
import boto3
from botocore.exceptions import ClientError
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings

# Configuration from environment
S3_ENDPOINT = os.getenv("S3_ENDPOINT", "http://minio:9000")
S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY", "minioadmin")
S3_SECRET_KEY = os.getenv("S3_SECRET_KEY", "minioadmin")
S3_REGION = os.getenv("S3_REGION", "us-east-1")
S3_BUCKET = os.getenv("S3_BUCKET", "claims")

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize MCP server with network settings
# Disable DNS rebinding protection to allow Kubernetes service DNS names
mcp = FastMCP(
    "s3-mcp-server",
    host="0.0.0.0",
    port=9097,
    transport_security=TransportSecuritySettings(
        allowed_hosts=os.getenv('ALLOWED_HOSTS', '*').split(',')
    )
)

# Initialize S3 client
s3_client = boto3.client(
    "s3",
    endpoint_url=S3_ENDPOINT,
    aws_access_key_id=S3_ACCESS_KEY,
    aws_secret_access_key=S3_SECRET_KEY,
    region_name=S3_REGION,
)


@mcp.tool()
def list_objects(prefix: str = "", bucket: str = S3_BUCKET) -> dict:
    """List objects in an S3 bucket with optional prefix filter.
    
    Args:
        prefix: Filter results to keys starting with this prefix
        bucket: S3 bucket name (defaults to configured bucket)
    
    Returns:
        Dictionary containing list of object keys and metadata
    """
    try:
        response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)
        objects = []
        for obj in response.get("Contents", []):
            objects.append({
                "key": obj["Key"],
                "size": obj["Size"],
                "last_modified": obj["LastModified"].isoformat()
            })
        return {"bucket": bucket, "prefix": prefix, "objects": objects}
    except ClientError as e:
        logger.error(f"Error listing objects: {e}")
        return {"error": str(e)}


@mcp.tool()
def get_object(key: str, bucket: str = S3_BUCKET) -> dict:
    """Retrieve an object from S3.
    
    Args:
        key: The object key (path) in the bucket
        bucket: S3 bucket name (defaults to configured bucket)
    
    Returns:
        Dictionary with object content (base64 encoded for binary) and metadata
    """
    try:
        response = s3_client.get_object(Bucket=bucket, Key=key)
        content = response["Body"].read()
        content_type = response.get("ContentType", "application/octet-stream")
        
        # Check if content is text or binary
        if content_type.startswith("text/") or content_type == "application/json":
            return {
                "key": key,
                "content": content.decode("utf-8"),
                "content_type": content_type,
                "encoding": "utf-8"
            }
        else:
            return {
                "key": key,
                "content": base64.b64encode(content).decode("utf-8"),
                "content_type": content_type,
                "encoding": "base64"
            }
    except ClientError as e:
        logger.error(f"Error getting object: {e}")
        return {"error": str(e)}


@mcp.tool()
def put_object(key: str, content: str, content_type: str = "text/plain", bucket: str = S3_BUCKET) -> dict:
    """Store an object in S3.
    
    Args:
        key: The object key (path) to store
        content: The content to store (string or base64 for binary)
        content_type: MIME type of the content
        bucket: S3 bucket name (defaults to configured bucket)
    
    Returns:
        Dictionary confirming the upload with object details
    """
    try:
        # Decode base64 if content is binary
        if content_type.startswith("image/") or content_type == "application/octet-stream":
            body = base64.b64decode(content)
        else:
            body = content.encode("utf-8")
        
        s3_client.put_object(
            Bucket=bucket,
            Key=key,
            Body=body,
            ContentType=content_type
        )
        return {
            "status": "success",
            "bucket": bucket,
            "key": key,
            "content_type": content_type
        }
    except ClientError as e:
        logger.error(f"Error putting object: {e}")
        return {"error": str(e)}


@mcp.tool()
def delete_object(key: str, bucket: str = S3_BUCKET) -> dict:
    """Delete an object from S3.
    
    Args:
        key: The object key (path) to delete
        bucket: S3 bucket name (defaults to configured bucket)
    
    Returns:
        Dictionary confirming the deletion
    """
    try:
        s3_client.delete_object(Bucket=bucket, Key=key)
        return {"status": "success", "deleted": key, "bucket": bucket}
    except ClientError as e:
        logger.error(f"Error deleting object: {e}")
        return {"error": str(e)}


if __name__ == "__main__":
    logger.info(f"Starting S3 MCP Server")
    logger.info(f"S3 Endpoint: {S3_ENDPOINT}")
    logger.info(f"Default Bucket: {S3_BUCKET}")
    
    # Run with streamable-http transport
    mcp.run(transport="streamable-http")