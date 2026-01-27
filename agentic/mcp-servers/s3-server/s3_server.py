from __future__ import annotations

import os
import base64
import logging
from typing import List, Literal, Optional
from typing_extensions import TypedDict

from aiobotocore.session import get_session
from botocore.config import Config
from botocore.exceptions import ClientError

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings


# =========================
# Configuration
# =========================

S3_ENDPOINT = os.getenv("S3_ENDPOINT", "http://minio:9000")
S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY", "minioadmin")
S3_SECRET_KEY = os.getenv("S3_SECRET_KEY", "minioadmin")
S3_REGION = os.getenv("S3_REGION", "us-east-1")

MAX_OBJECT_SIZE_MB = int(os.getenv("MAX_OBJECT_SIZE_MB", "50"))
MAX_OBJECT_BYTES = MAX_OBJECT_SIZE_MB * 1024 * 1024
AUTO_CREATE_BUCKET = os.getenv("AUTO_CREATE_BUCKET", "true").lower() == "true"

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "9097"))


# =========================
# Logging
# =========================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mcp-s3-server")


# =========================
# MCP Server
# =========================

mcp = FastMCP(
    name="mcp-s3-server",
    stateless_http=True,
    json_response=True,
    transport_security=TransportSecuritySettings(
        allowed_hosts=os.getenv("ALLOWED_HOSTS", "*").split(",")
    ),
)


# =========================
# S3 Client (async)
# =========================

_boto_config = Config(
    retries={"max_attempts": 5, "mode": "standard"},
    connect_timeout=5,
    read_timeout=60,
)

_session = get_session()


async def get_s3():
    return _session.create_client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=S3_ACCESS_KEY,
        aws_secret_access_key=S3_SECRET_KEY,
        region_name=S3_REGION,
        config=_boto_config,
    )


# =========================
# Typed Responses
# =========================

class ErrorResponse(TypedDict):
    error: str
    code: str


class ObjectInfo(TypedDict):
    key: str
    size: int
    last_modified: str


class ListObjectsResponse(TypedDict):
    bucket: str
    prefix: str
    objects: List[ObjectInfo]


class GetObjectResponse(TypedDict):
    key: str
    content: str
    content_type: str
    encoding: Literal["utf-8", "base64"]
    etag: str


class StatusResponse(TypedDict):
    status: Literal["success"]
    bucket: str
    key: str


class BucketInfo(TypedDict):
    name: str
    creation_date: str


class ListBucketsResponse(TypedDict):
    buckets: List[BucketInfo]


# =========================
# MCP Tools
# =========================

@mcp.tool(description="List all available S3 buckets.")
async def list_buckets() -> ListBucketsResponse | ErrorResponse:
    try:
        async with await get_s3() as s3:
            response = await s3.list_buckets()
            buckets: List[BucketInfo] = []
            
            for bucket in response.get("Buckets", []):
                buckets.append(
                    {
                        "name": bucket["Name"],
                        "creation_date": bucket["CreationDate"].isoformat(),
                    }
                )
            
            return {"buckets": buckets}
    
    except ClientError as e:
        return {"error": str(e), "code": "S3_LIST_BUCKETS_FAILED"}


@mcp.tool(description="List objects in an S3 bucket with optional prefix filter.")
async def list_objects(prefix: str = "", bucket: str = "") -> ListObjectsResponse | ErrorResponse:
    bucket = bucket
    try:
        async with await get_s3() as s3:
            paginator = s3.get_paginator("list_objects_v2")
            objects: List[ObjectInfo] = []

            async for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
                for obj in page.get("Contents", []):
                    objects.append(
                        {
                            "key": obj["Key"],
                            "size": obj["Size"],
                            "last_modified": obj["LastModified"].isoformat(),
                        }
                    )

            return {"bucket": bucket, "prefix": prefix, "objects": objects}

    except ClientError as e:
        return {"error": str(e), "code": "S3_LIST_FAILED"}


@mcp.tool(description="Retrieve an object from S3.")
async def get_object(key: str, bucket: str = "") -> GetObjectResponse | ErrorResponse:
    bucket = bucket
    try:
        async with await get_s3() as s3:
            head = await s3.head_object(Bucket=bucket, Key=key)

            if head["ContentLength"] > MAX_OBJECT_BYTES:
                return {
                    "error": "Object exceeds maximum allowed size",
                    "code": "OBJECT_TOO_LARGE",
                }

            resp = await s3.get_object(Bucket=bucket, Key=key)
            body = await resp["Body"].read()
            content_type = resp.get("ContentType", "application/octet-stream")

            if content_type.startswith("text/") or content_type == "application/json":
                try:
                    return {
                        "key": key,
                        "content": body.decode("utf-8"),
                        "content_type": content_type,
                        "encoding": "utf-8",
                        "etag": head["ETag"],
                    }
                except UnicodeDecodeError:
                    pass

            return {
                "key": key,
                "content": base64.b64encode(body).decode(),
                "content_type": content_type,
                "encoding": "base64",
                "etag": head["ETag"],
            }

    except ClientError as e:
        return {"error": str(e), "code": "S3_GET_FAILED"}


@mcp.tool(description="Store an object in S3.")
async def put_object(
    key: str,
    content: str,
    content_type: str = "text/plain",
    bucket: str = "",
    if_match: Optional[str] = None,
) -> StatusResponse | ErrorResponse:
    bucket = bucket
    try:
        body = (
            base64.b64decode(content)
            if content_type.startswith(("image/", "application/octet-stream"))
            else content.encode("utf-8")
        )

        if len(body) > MAX_OBJECT_BYTES:
            return {"error": "Upload exceeds size limit", "code": "OBJECT_TOO_LARGE"}

        async with await get_s3() as s3:
            kwargs = {
                "Bucket": bucket,
                "Key": key,
                "Body": body,
                "ContentType": content_type,
            }

            if if_match:
                kwargs["IfMatch"] = if_match

            await s3.put_object(**kwargs)

        return {"status": "success", "bucket": bucket, "key": key}

    except ClientError as e:
        return {"error": str(e), "code": "S3_PUT_FAILED"}


@mcp.tool(description="Delete an object from S3.")
async def delete_object(key: str, bucket: str = "") -> StatusResponse | ErrorResponse:
    bucket = bucket
    try:
        async with await get_s3() as s3:
            await s3.delete_object(Bucket=bucket, Key=key)
        return {"status": "success", "bucket": bucket, "key": key}

    except ClientError as e:
        return {"error": str(e), "code": "S3_DELETE_FAILED"}


# Add health check endpoint using custom_route decorator
@mcp.custom_route("/health", methods=["GET"])
async def health_check(request):
    """Health check endpoint for Kubernetes probes"""
    from starlette.responses import JSONResponse
    return JSONResponse({
        "status": "healthy"
    })

# =========================
# Entrypoint
# =========================

if __name__ == "__main__":
    import uvicorn

    app = mcp.streamable_http_app()

    uvicorn.run(
        app,
        host=HOST,
        port=PORT,
        log_level="info",
        access_log=True,
    )