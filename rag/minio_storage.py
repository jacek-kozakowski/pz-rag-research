import boto3
import os
import tempfile
from botocore.client import Config

MINIO_BUCKET_NAME = os.environ.get("MINIO_BUCKET_NAME", "rag-bucket")

def get_minio_client():
    return boto3.client(
        "s3",
        endpoint_url=os.environ.get("MINIO_ENDPOINT", "http://localhost:9000"),
        aws_access_key_id=os.environ.get("MINIO_ACCESS_KEY"),
        aws_secret_access_key=os.environ.get("MINIO_SECRET_KEY"),
        config=Config(signature_version="s3v4"),
    )

def ensure_bucket():
    client = get_minio_client()
    try:
        client.head_bucket(Bucket=MINIO_BUCKET_NAME)
    except Exception:
        client.create_bucket(Bucket=MINIO_BUCKET_NAME)
        print(f"Created bucket {MINIO_BUCKET_NAME}")

def upload_file(file_path: str, filename: str) -> str:
    ensure_bucket()
    client = get_minio_client()
    try:
        client.upload_file(file_path, MINIO_BUCKET_NAME, filename)
        return filename
    except Exception as e:
        print(f"Failed to upload {file_path} to {MINIO_BUCKET_NAME}: {e}")
        return ""

def upload_bytes(data: bytes, filename: str) -> str:
    ensure_bucket()
    client = get_minio_client()
    try:
        import io
        client.upload_fileobj(io.BytesIO(data), MINIO_BUCKET_NAME, filename)
        return filename
    except Exception as e:
        print(f"Failed to upload {filename} to {MINIO_BUCKET_NAME}: {e}")

def list_files() -> list[str]:
    ensure_bucket()
    client = get_minio_client()
    response = client.list_objects_v2(Bucket=MINIO_BUCKET_NAME)
    return [obj["Key"] for obj in response.get("Contents", [])]


def download_to_temp(filename: str) -> str:
    ensure_bucket()
    client = get_minio_client()
    ext = os.path.splitext(filename)[-1]
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        client.download_fileobj(MINIO_BUCKET_NAME, filename, tmp)
        return tmp.name

def delete_file(filename: str):
    ensure_bucket()
    client = get_minio_client()
    client.delete_object(Bucket=MINIO_BUCKET_NAME, Key=filename)