# src/utils/config_loader.py

import os
from dotenv import load_dotenv

# --- Load the .env file ---
# This line finds the .env file in your project root and loads its content
# into the system's environment variables, ready for use.
load_dotenv()

# --- Getter functions for your secret keys ---

def get_serpapi_key():
    """
    Safely retrieves the SerpApi API key from the environment.
    Returns:
        str: The API key, or None if it's not found.
    """
    return os.getenv("SERPAPI_KEY")

def get_reddit_client_id():
    """
    Safely retrieves the Reddit Client ID from the environment.
    Returns:
        str: The Client ID, or None if it's not found.
    """
    return os.getenv("REDDIT_CLIENT_ID")

def get_reddit_client_secret():
    """
    Safely retrieves the Reddit Client Secret from the environment.
    Returns:
        str: The Client Secret, or None if it's not found.
    """
    return os.getenv("REDDIT_CLIENT_SECRET")

def get_reddit_user_agent():
    """
    Safely retrieves the Reddit User Agent from the environment.
    Returns:
        str: The User Agent string, or None if it's not found.
    """
    return os.getenv("REDDIT_USER_AGENT")

# You can add more functions here as you add more keys to your .env file.
# For example:
#
# def get_twitter_bearer_token():
#     return os.getenv("TWITTER_BEARER_TOKEN")