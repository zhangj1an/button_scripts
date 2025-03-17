import sys 
import os 
import json 
import copy
import re 
sys.path.append(os.path.expanduser("~/TextToActions/code"))
from simulated.utils.extract_python_code import extract_python_code

def matches_run_action_format(text):
    """
    Check if the provided text matches the specified run_action format.

    Args:
    text (str): The text to check against the regex pattern.

    Returns:
    bool: True if the text matches the regex pattern, False otherwise.
    """
    # can extract from the text
    text = extract_python_code(text)
    pattern = r"^(run_action\(\s*(['\"])([^'\"]+)\2\s*,\s*execution_times\s*=\s*(\d+)(\s*,\s*duration\s*=\s*(\d+))?\s*\)$|end)"
    return re.match(pattern, text) is not None

if __name__ == "__main__":
    # Test the function with example text
    text = "run_action('turn_power_selector_dial_clockwise', execution_times=4)"
    print(matches_run_action_format(text))  # Expected output: True
    exit()
    text = "run_action('heat_coffee', execution_times=1, duration=60)"
    print(matches_run_action_format(text))  # Expected output: True

    text = "run_action('heat_coffee', execution_times=1)"
    print(matches_run_action_format(text))  # Expected output: True

    text = "run_action('heat_coffee', execution_times=1, duration=60)  # Comment"
    print(matches_run_action_format(text))  # Expected output: True

    text = "run_action('heat_coffee', execution_times=1, duration=60, extra_param=123)"
    print(matches_run_action_format(text))  # Expected output: False

    text = "end"
    print(matches_run_action_format(text))  # Expected output: True