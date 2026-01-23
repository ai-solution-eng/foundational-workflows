import boto3
from botocore.exceptions import ClientError
import os
import urllib3
from utils import update_auth_token

os.environ["AWS_REQUEST_CHECKSUM_CALCULATION"] = "WHEN_REQUIRED"
os.environ["AWS_RESPONSE_CHECKSUM_VALIDATION"] = "WHEN_REQUIRED"
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def object_exists(s3_client, args):
    try:
        s3_client.head_object(
            Bucket=args.s3_bucket_name,
            Key="model/" + args.bento_file.split("/")[-1],
        )
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] == "404":
            return False
        raise


def create_s3_bucket(args):
    s3_client = boto3.client(
        "s3",
        aws_access_key_id=args.aws_access_key_id,
        aws_secret_access_key=args.aws_secret_access_key,
        endpoint_url=args.endpoint_url,
        verify=False,
    )
    print("Buckets before creation:")
    print(s3_client.list_buckets()["Buckets"])

    try:
        s3_client.create_bucket(Bucket=args.s3_bucket_name)
    except s3_client.exceptions.BucketAlreadyOwnedByYou:
        pass

    print("Buckets after creation:")
    print(s3_client.list_buckets()["Buckets"])
    return s3_client


def upload_file_to_s3(s3_client, args):
    try:
        s3_client.upload_file(
            Bucket=args.s3_bucket_name,
            Filename=args.bento_file,
            Key="model/" + args.bento_file.split("/")[-1],
        )

        print(
            f"""File {args.bento_file.split("/")[-1]} 
            uploaded to bucket {args.s3_bucket_name} as 
            {"model/" + args.bento_file.split("/")[-1]}
            """
        )
    except ClientError as e:
        print(f"Error uploading file: {e}")


if __name__ == "__main__":
    import argparse
    import boto3
    from botocore.exceptions import ClientError

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--s3-bucket-name",
        type=str,
        help=" Name of the S3 bucket to create or use",
    )
    parser.add_argument(
        "--aws-access-key-id",
        type=str,
        help="AWS Access Key ID",
    )
    parser.add_argument(
        "--aws-secret-access-key",
        type=str,
        help="AWS Secret Access Key",
    )
    parser.add_argument(
        "--endpoint-url",
        type=str,
        help="Custom S3 Endpoint URL",
    )
    parser.add_argument(
        "--bento-file",
        type=str,
        help="Path to the BentoML model file to upload",
    )

    args = parser.parse_args()

    update_auth_token()
    s3_client = create_s3_bucket(args)
    upload_file_to_s3(s3_client, args)
    print(object_exists(s3_client, args))
