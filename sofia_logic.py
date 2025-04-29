import os
import time
import tempfile
import boto3
import gc
import importlib
import json
from dotenv import load_dotenv
from services.Intranet_repository_s3 import IntranetRepository
from chains import first_responder, final_responder, global_responder, vid_responder
from langgraph.graph import MessageGraph
from classes import FinalResponse
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage


# Create the processing graph for the LLM
def create_graph():
    builder = MessageGraph()
    builder.add_node("classifier", first_responder)
    builder.add_node("global", global_responder)
    builder.add_node("vendorid", vid_responder)
    builder.add_node("final", final_responder)
    builder.add_conditional_edges("classifier", decision_flow)
    builder.add_edge("global", "final")
    builder.add_edge("vendorid", "final")
    builder.set_entry_point("classifier")
    return builder.compile()

def decision_flow(state: list[BaseMessage]) -> str:
    last_message = state[-1]
    if hasattr(last_message, 'additional_kwargs') and 'tool_calls' in last_message.additional_kwargs:
        tool_calls = last_message.additional_kwargs['tool_calls']
        if tool_calls:
            last_tool = tool_calls[-1]
            arguments = last_tool['function']['arguments']
            result = json.loads(arguments)
            if result.get("request_type") == "global_question":
                return "global"
            elif result.get("request_type") == "vendorid":
                return "vendorid"
    return "final"

# AWS configuration function
def configure_aws():
    """
    Configure AWS credentials from environment variables.
    """
    load_dotenv()
    
    # Check if AWS credentials are set in environment variables
    aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')
    aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
    aws_region = os.getenv('AWS_REGION', 'us-east-1')
    
    if not aws_access_key or not aws_secret_key:
        raise ValueError(
            "AWS credentials not found. Please set AWS_ACCESS_KEY_ID and "
            "AWS_SECRET_ACCESS_KEY environment variables."
        )
    
    # Configure boto3 session
    boto3.setup_default_session(
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_key,
        region_name=aws_region
    )
    
    print(f"AWS configured with region: {aws_region}")
    
    return {
        'region': aws_region,
        'credentials_found': True
    }

# Function to upload file to S3
def upload_file_to_s3(bucket_name, file_object, object_name=None):
    """
    Upload a file to an S3 bucket
    
    :param bucket_name: Bucket to upload to
    :param file_object: File-like object to upload
    :param object_name: S3 object name. If not specified then the file name is used
    :return: True if file was uploaded, else False
    """
    # If object name not specified, use the file name
    if object_name is None:
        object_name = file_object.name
        
    # Upload the file
    s3_client = boto3.client('s3')
    try:
        # Create a temporary file to save the uploaded file content
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(file_object.getvalue())
            temp_file_path = temp_file.name
        
        # Upload file from disk to S3
        s3_client.upload_file(temp_file_path, bucket_name, object_name)
        
        # Remove temporary file
        os.remove(temp_file_path)
        
        return True
    except Exception as e:
        print(f"Error uploading file to S3: {e}")
        return False

# Function to delete file from S3 directly - simplified approach
def delete_file_direct(bucket_name, file_key):
    """
    Direct method to delete a file from S3, with minimal complications
    
    :param bucket_name: Bucket to delete from
    :param file_key: S3 object key to delete
    :return: True if file was deleted, else False
    """
    try:
        # Initialize boto3 session and client directly
        session = boto3.Session(
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            region_name=os.getenv('AWS_REGION', 'us-east-1')
        )
        
        s3 = session.resource('s3')
        bucket = s3.Bucket(bucket_name)
        obj = bucket.Object(file_key)
        
        # Delete the object directly
        response = obj.delete()
        
        # Log the response
        print(f"Delete response: {response}")
        
        # Check if deletion was successful based on response status
        if response['ResponseMetadata']['HTTPStatusCode'] == 204:
            print(f"Successfully deleted {file_key}")
            return True
        else:
            print(f"Deletion may have failed. Response: {response}")
            return False
            
    except Exception as e:
        print(f"Error in delete_file_direct: {e}")
        import traceback
        traceback.print_exc()
        return False

# Function to view S3 file content
def view_s3_file_content(bucket_name, file_key):
    """
    Retrieve and display content of a file from S3
    """
    s3_client = boto3.client('s3')
    try:
        # Download file from S3
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            s3_client.download_file(bucket_name, file_key, temp_file.name)
            temp_file_path = temp_file.name
        
        # Read file content
        with open(temp_file_path, 'r') as f:
            content = f.read()
        
        # Clean up temporary file
        os.remove(temp_file_path)
        
        return content
    except Exception as e:
        print(f"Error retrieving file from S3: {e}")
        return f"Error retrieving file: {str(e)}"

# Get repository instance
def get_repository(bucket_name="docs-intranet"):
    """
    Get a repository instance for the specified bucket
    """
    try:
        repository = IntranetRepository(bucket_name=bucket_name)
        vectorstore = repository.create_or_load_faiss_index()
        return repository, vectorstore
    except Exception as e:
        print(f"Error creating repository: {e}")
        return None, None

# Force reindex all documents
def force_reindex(bucket_name="docs-intranet"):
    """
    Force complete reindexing of ALL documents from S3 bucket
    Returns: repository, vectorstore, elapsed_time, doc_count
    """
    try:
        # Clear IntranetRepository singleton
        IntranetRepository._instance = None
        IntranetRepository._vectorstore = None
        
        # Clear module-level references in chains.py
        import chains
        if hasattr(chains, 'intranet_repository'):
            chains.intranet_repository = None
        if hasattr(chains, 'vectorstore'):
            chains.vectorstore = None
        
        # Clear physical index files
        index_path = "faiss_index"
        index_files = [
            os.path.join(index_path, "index.faiss"),
            os.path.join(index_path, "index.pkl")
        ]
        
        for file_path in index_files:
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except Exception as e:
                    print(f"Error removing {file_path}: {e}")
        
        # Check if directory exists and is empty, then remove it
        if os.path.exists(index_path) and os.path.isdir(index_path):
            if not os.listdir(index_path):  # If directory is empty
                try:
                    os.rmdir(index_path)
                except Exception as e:
                    print(f"Error removing directory {index_path}: {e}")
        
        # List ALL documents in bucket
        s3_client = boto3.client('s3')
        response = s3_client.list_objects_v2(Bucket=bucket_name)
        
        doc_count = 0
        if 'Contents' in response:
            doc_count = len(response['Contents'])
            
        # Create a completely isolated new repository instance
        class FullIndexRepository(IntranetRepository):
            def list_documents_in_bucket(self):
                """Override: List ALL documents in S3 without filtering."""
                try:
                    response = self.s3_client.list_objects_v2(Bucket=self.bucket_name)
                    if 'Contents' in response:
                        all_files = [item['Key'] for item in response['Contents']]
                        return all_files
                    return []
                except Exception as e:
                    print(f"Error listing objects in S3 bucket: {e}")
                    return []
        
        # Use specialized class for reindexing
        start_time = time.time()
        new_repository = FullIndexRepository(bucket_name=bucket_name)
        
        # Force index rebuild, ensuring cache is not used
        vectorstore = new_repository.force_rebuild_index()
        elapsed_time = time.time() - start_time
        
        # Update module references
        import chains
        chains.intranet_repository = new_repository
        chains.vectorstore = vectorstore
        
        return new_repository, vectorstore, elapsed_time, doc_count
    except Exception as e:
        print(f"Error during reindexing: {e}")
        return None, None, 0, 0

# Memory cleanup function
def cleanup_memory():
    """
    Function to clean up all in-memory references while preserving history
    """
    try:
        # Reset the IntranetRepository singleton
        IntranetRepository._instance = None
        IntranetRepository._vectorstore = None
                
        # Reset module-level variables in chains.py
        import chains
        
        # Reset specific module variables
        if hasattr(chains, 'intranet_repository'):
            chains.intranet_repository = None
        if hasattr(chains, 'vectorstore'):
            chains.vectorstore = None
            
        # Force garbage collection to clean up lingering references
        gc.collect()
        
        # Verify FAISS index directory exists
        index_path = "faiss_index"
        os.makedirs(index_path, exist_ok=True)
        
        return True
    except Exception as e:
        print(f"Error during memory cleanup: {e}")
        return False

# Get list of documents in S3 bucket
def list_s3_documents(bucket_name="docs-intranet"):
    """Get list of documents in S3 bucket"""
    try:
        s3_client = boto3.client('s3')
        response = s3_client.list_objects_v2(Bucket=bucket_name)
        
        if 'Contents' in response:
            return [item for item in response['Contents']]
        return []
    except Exception as e:
        print(f"Error listing S3 documents: {e}")
        return []