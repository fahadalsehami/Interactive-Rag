"""
model.py: Utilities for model parameters configuration and Kendra integration
"""
from typing import Dict, Any, Optional
import boto3
from botocore.exceptions import ClientError
import logging

logger = logging.getLogger(__name__)

# Define constants
CLAUDE_MODEL_ID = "anthropic.claude-3-sonnet-20240229-v1:0"


def get_kendra_config() -> Optional[Dict[str, str]]:
    """
    Retrieve Kendra configuration from the CloudFormation stack.
    Returns:
        Optional[Dict[str, str]]: Configuration with `index_id` and `region`.
    """
    try:
        import os
        # Dynamically fetch stack name and region
        stack_name = os.getenv('KENDRA_STACK_NAME', 'clinical-kendra-stack-20241221')
        region_name = os.getenv('AWS_REGION', boto3.session.Session().region_name)

        cfn = boto3.client('cloudformation', region_name=region_name)
        response = cfn.describe_stacks(StackName=stack_name)

        # Ensure stack data exists
        stacks = response.get('Stacks', [])
        if not stacks:
            logger.error(f"No stack data found for '{stack_name}'")
            return None

        # Extract outputs
        outputs = {o['OutputKey']: o['OutputValue'] for o in stacks[0].get('Outputs', [])}
        kendra_index_id = outputs.get('KendraIndexID')

        if not kendra_index_id:
            logger.error("No Kendra Index ID found in stack outputs")
            return None

        config = {'index_id': kendra_index_id, 'region': region_name}
        logger.info(f"Retrieved Kendra config: {config}")
        return config

    except ClientError as e:
        error_code = e.response['Error'].get('Code')
        if error_code == 'ValidationError':
            logger.info(f"Kendra stack '{stack_name}' not found - continuing without Kendra integration")
        else:
            logger.error(f"AWS error: {e}")
        return None

    except Exception as e:
        logger.error(f"Unexpected error retrieving Kendra config: {e}")
        return None

def validate_kendra_config(kendra_config: Optional[Dict[str, str]]) -> bool:
    """
    Validate if the retrieved Kendra configuration is complete.

    Args:
        kendra_config (Optional[Dict[str, str]]): The Kendra configuration to validate.

    Returns:
        bool: True if valid, False otherwise.
    """
    if not kendra_config:
        logger.error("Kendra configuration is missing or incomplete.")
        return False
    required_keys = ["index_id", "region"]
    for key in required_keys:
        if key not in kendra_config:
            logger.error(f"Kendra configuration missing required key: {key}")
            return False
    return True
    """
    Retrieve Kendra configuration from the CloudFormation stack.
    Returns:
        Optional[Dict[str, str]]: Configuration with `index_id` and `region`.
    """
    try:
        cfn = boto3.client('cloudformation')
        stack_name = 'clinical-kendra-index'
        response = cfn.describe_stacks(StackName=stack_name)

        # Ensure stack data exists
        stacks = response.get('Stacks', [])
        if not stacks:
            logger.error("No stack data found for 'clinical-kendra-index'")
            return None

        # Extract outputs
        outputs = {o['OutputKey']: o['OutputValue'] for o in stacks[0].get('Outputs', [])}
        kendra_index_id = outputs.get('KendraIndexID')
        aws_region = outputs.get('AWSRegion', boto3.session.Session().region_name)

        if not kendra_index_id:
            logger.error("No Kendra Index ID found in stack outputs")
            return None

        config = {'index_id': kendra_index_id, 'region': aws_region}
        logger.info(f"Retrieved Kendra config: {config}")
        return config

    except ClientError as e:
        error_code = e.response['Error'].get('Code')
        if error_code == 'ValidationError':
            logger.info("Kendra stack not found - continuing without Kendra integration")
        else:
            logger.error(f"AWS error: {e}")
        return None

    except Exception as e:
        logger.error(f"Unexpected error retrieving Kendra config: {e}")
        return None


def get_model_params(
    model_id: str = CLAUDE_MODEL_ID,
    temperature: float = 0.9,
    max_tokens: int = 4096
) -> Dict[str, Any]:
    """
    Get model parameters for Claude 3 Sonnet.

    Args:
        model_id (str): Model identifier.
        temperature (float): Temperature parameter (0-1).
        max_tokens (int): Maximum tokens in response.

    Returns:
        Dict[str, Any]: Model parameters.
    """
    if model_id != CLAUDE_MODEL_ID:
        logger.warning(f"Using non-standard model ID: {model_id}")

    return {
        "model": model_id,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": 0.9,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "anthropic_version": "bedrock-2024-02-29",
        "messages": []
    }


def validate_aws_credentials() -> bool:
    """
    Validate AWS credentials are properly configured.

    Returns:
        bool: True if credentials are valid, False otherwise.
    """
    try:
        sts = boto3.client('sts')
        identity = sts.get_caller_identity()
        logger.info(f"AWS credentials valid for account: {identity['Account']}")
        return True
    except Exception as e:
        logger.error(f"AWS credentials validation failed: {e}")
        return False


def get_bedrock_config() -> Dict[str, Any]:
    """
    Get AWS Bedrock configuration.

    Returns:
        Dict[str, Any]: Bedrock configuration.
    """
    return {
        "model_id": CLAUDE_MODEL_ID,
        "provider": "anthropic",
        "service_name": "bedrock"
    }
