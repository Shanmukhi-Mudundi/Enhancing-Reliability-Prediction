"""Configuration management module."""

import os
import yaml
from pathlib import Path
from dotenv import load_dotenv


def load_config(config_file: str = "config.yaml") -> dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_file: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_env_vars():
    """
    Load environment variables from .env file.
    """
    load_dotenv()
    
    env_vars = {
        'together_api_key': os.getenv('TOGETHER_API_KEY'),
        'together_model': os.getenv('TOGETHER_MODEL', 'mistralai/Mixtral-8x7B-Instruct-v0.1'),
        'together_api_url': os.getenv('TOGETHER_API_URL', 'https://api.together.xyz/v1/chat/completions'),
    }
    
    if not env_vars['together_api_key']:
        raise ValueError("TOGETHER_API_KEY environment variable not set. Please create a .env file.")
    
    return env_vars
