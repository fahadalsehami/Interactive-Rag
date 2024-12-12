"""
model.py: Utilities for model parameters configuration and Kendra integration
"""
from typing import Dict, Any, Optional
import boto3
from botocore.exceptions import ClientError
import logging
import os

logger = logging.getLogger(__name__)

# Define constants
CLAUDE_MODEL_ID = "anthropic.claude-3-sonnet-20240229-v1:0"

def get_kendra_config() -> Optional[Dict[str, str]]:
    """
    Get Kendra configuration from CloudFormation stack.
    
    Returns:
        Optional[Dict[str, str]]: Configuration with index_id and region
    """
    try:
        cfn = boto3.client('cloudformation')
        
        # List all stacks and find Kendra stack
        paginator = cfn.get_paginator('list_stacks')
        kendra_stack = None
        
        for page in paginator.paginate():
            for stack in page['StackSummaries']:
                if stack['StackStatus'] in ['CREATE_COMPLETE', 'UPDATE_COMPLETE']:
                    if 'kendra' in stack['StackName'].lower():
                        kendra_stack = stack['StackName']
                        logger.info(f"Found Kendra stack: {kendra_stack}")
                        break
        
        if not kendra_stack:
            logger.warning("No active Kendra stack found")
            return None
            
        # Get stack outputs from AWS
        response = cfn.describe_stacks(StackName=kendra_stack)
        
        # Verify we have stack data
        if not response.get('Stacks'):
            logger.error("No stack data found")
            return None
            
        # Get stack outputs
        stack = response['Stacks'][0]
        if 'Outputs' not in stack:
            logger.error(f"No outputs found in stack {kendra_stack}")
            return None
            
        # Process outputs
        outputs = {}
        for output in stack['Outputs']:
            outputs[output['OutputKey']] = output['OutputValue']
        
        # Get current region
        session = boto3.session.Session()
        current_region = session.region_name
        
        # Create config with required AWS Kendra values
        config = {
            'index_id': outputs.get('KendraIndexID'),
            'region': outputs.get('AWSRegion', current_region)
        }
        
        # Validate config
        if not config['index_id']:
            logger.error("No Kendra Index ID found in stack outputs")
            return None
            
        logger.info(f"Retrieved Kendra config: {config}")
        return config
        
    except ClientError as e:
        if e.response['Error']['Code'] == 'ValidationError':
            logger.info("Kendra stack not found - continuing without Kendra integration")
            return None
        else:
            logger.error(f"AWS error: {str(e)}")
            return None
    except Exception as e:
        logger.error(f"Error getting Kendra config: {str(e)}")
        return None

def get_model_params(
    model_id: str = CLAUDE_MODEL_ID,
    temperature: float = 0.9,
    max_tokens: int = 4096
) -> Dict[str, Any]:
    """
    Get model parameters for Claude 3 Sonnet.
    
    Args:
        model_id (str): Model identifier (defaults to Claude 3 Sonnet)
        temperature (float): Temperature parameter (0-1)
        max_tokens (int): Maximum tokens in response
        
    Returns:
        Dict[str, Any]: Model parameters
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
        bool: True if credentials are valid
    """
    try:
        sts = boto3.client('sts')
        identity = sts.get_caller_identity()
        logger.info(f"AWS credentials valid for account: {identity['Account']}")
        return True
    except Exception as e:
        logger.error(f"AWS credentials validation failed: {str(e)}")
        return False

def get_bedrock_config() -> Dict[str, Any]:
    """
    Get AWS Bedrock configuration.
    
    Returns:
        Dict[str, Any]: Bedrock configuration
    """
    return {
        "model_id": CLAUDE_MODEL_ID,
        "provider": "anthropic",
        "service_name": "bedrock"
    }