import logging
import pprint
import sys
from pathlib import Path

# This ensures the 'src' directory is in the Python path
# so we can import our parser.
project_root = Path(__file__).parent
src_path = project_root / 'src'
sys.path.insert(0, str(project_root))

from src.parser.command_parser import CommandParser

# Set up logging to see warnings from the parser
logging.basicConfig(level=logging.INFO)
print(f"--- Python is using this interpreter: {sys.executable} ---")
print(f"--- Attempting to import CommandParser from: {src_path} ---")


print("\n--- Creating CommandParser instance ---")
try:
    parser = CommandParser()
    print("--- Parser created successfully ---")
except Exception as e:
    print(f"--- FAILED to create parser: {e} ---")
    # If it fails here, something is very wrong with the file.
    exit()

prompt = "a dataset of happy dogs vs. sad dogs,"
print(f"\n--- Parsing prompt: '{prompt}' ---")

result = parser.parse(prompt)

print("\n--- FULL PARSER RESULT ---")
pprint.pprint(result)

print("\n--- Specifically, 'class_labels' is: ---")
# This is the value that the test is checking.
labels = result.get("class_labels")
print(labels)
print(f"--- The type of 'class_labels' is: {type(labels).name} ---")