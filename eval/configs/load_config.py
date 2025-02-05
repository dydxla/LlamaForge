import yaml
import os

def read_yaml_to_dict(file_path: str) -> dict:
    """
    Reads a YAML file and converts it to a Python dictionary.

    Args:
        file_path (str): The path to the YAML file.

    Returns:
        dict: A dictionary representation of the YAML file content.
    """
    try:
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)  # Parse the YAML file
            return data
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
        return {}
    except yaml.YAMLError as e:
        print(f"Error: Failed to parse YAML file. {e}")
        return {}
    
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
config_path = os.path.join(current_dir, 'config.yaml')
config = read_yaml_to_dict(config_path)