"""
model.py: AWS configuration, Amazon Kendra lookup and Claude on Amazon Bedrock
"""
import logging
import os
from typing import Dict, Optional

import boto3
from botocore.exceptions import BotoCoreError, ClientError

logger = logging.getLogger(__name__)

# Claude on Amazon Bedrock. Set BEDROCK_MODEL_ID to use another model ID or inference profile
# that your account can invoke, for example us.anthropic.claude-sonnet-4-6 to keep inference in the US.
DEFAULT_MODEL_ID = "global.anthropic.claude-sonnet-4-6"

# Name of the CloudFormation stack created from kendra-template.yaml
DEFAULT_STACK_NAME = "clinical-rag-kendra"

DEFAULT_REGION = "us-east-1"


def model_id() -> str:
    """Return the Bedrock model ID or inference profile ID to call."""
    return os.getenv("BEDROCK_MODEL_ID") or DEFAULT_MODEL_ID


def aws_region() -> str:
    """Return the AWS Region from AWS_REGION, then the AWS configuration, then the default."""
    return os.getenv("AWS_REGION") or boto3.session.Session().region_name or DEFAULT_REGION


def validate_aws_credentials() -> bool:
    """
    Check that AWS credentials are configured and valid.

    Returns:
        bool: True if credentials are valid, False otherwise.
    """
    if boto3.session.Session().get_credentials() is None:
        logger.info("No AWS credentials found - running on the RDoC matrix alone")
        return False
    try:
        identity = boto3.client("sts", region_name=aws_region()).get_caller_identity()
        logger.info(f"AWS credentials valid for account: {identity['Account']}")
        return True
    except (BotoCoreError, ClientError) as e:
        logger.warning(f"AWS credentials could not be validated: {e}")
        return False


def get_knowledge_base_config() -> Optional[Dict[str, str]]:
    """
    Find the Amazon Bedrock Knowledge Base to retrieve from, set by KNOWLEDGE_BASE_ID.

    Returns:
        Optional[Dict[str, str]]: Configuration with `knowledge_base_id` and `region`, or None.
    """
    knowledge_base_id = os.getenv("KNOWLEDGE_BASE_ID")
    if not knowledge_base_id:
        return None
    region = aws_region()
    logger.info(f"Using Bedrock Knowledge Base {knowledge_base_id} in {region}")
    return {"knowledge_base_id": knowledge_base_id, "region": region}


def get_kendra_config() -> Optional[Dict[str, str]]:
    """
    Find the Kendra index to retrieve from.

    Uses KENDRA_INDEX_ID when it is set; otherwise reads the KendraIndexID output of the
    CloudFormation stack named by KENDRA_STACK_NAME.

    Returns:
        Optional[Dict[str, str]]: Configuration with `index_id` and `region`, or None.
    """
    region = aws_region()
    index_id = os.getenv("KENDRA_INDEX_ID")
    if index_id:
        return {"index_id": index_id, "region": region}

    stack_name = os.getenv("KENDRA_STACK_NAME") or DEFAULT_STACK_NAME
    try:
        cfn = boto3.client("cloudformation", region_name=region)
        stacks = cfn.describe_stacks(StackName=stack_name).get("Stacks", [])
    except ClientError as e:
        if e.response["Error"].get("Code") == "ValidationError":
            logger.info(f"Kendra stack '{stack_name}' not found - continuing without retrieval")
        else:
            logger.error(f"AWS error: {e}")
        return None
    except BotoCoreError as e:
        logger.error(f"AWS error: {e}")
        return None

    if not stacks:
        logger.error(f"No stack data found for '{stack_name}'")
        return None

    outputs = {o["OutputKey"]: o["OutputValue"] for o in stacks[0].get("Outputs", [])}
    index_id = outputs.get("KendraIndexID")
    if not index_id:
        logger.error("No Kendra Index ID found in stack outputs")
        return None

    logger.info(f"Using Kendra index {index_id} in {region}")
    return {"index_id": index_id, "region": region}


def generate_text(prompt: str, system: str, max_tokens: int = 1500, temperature: float = 0.2) -> str:
    """
    Send one prompt to Claude on Amazon Bedrock with the Converse API.

    Args:
        prompt (str): User message.
        system (str): System prompt.
        max_tokens (int): Maximum tokens in the reply.
        temperature (float): Sampling temperature (0-1).

    Returns:
        str: Text of the reply.
    """
    client = boto3.client("bedrock-runtime", region_name=aws_region())
    response = client.converse(
        modelId=model_id(),
        system=[{"text": system}],
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={"maxTokens": max_tokens, "temperature": temperature},
    )
    content = response["output"]["message"]["content"]
    return "".join(block.get("text", "") for block in content).strip()
