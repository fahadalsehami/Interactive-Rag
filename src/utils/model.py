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
