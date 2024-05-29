import json


def load_static_file(file_path: str):
    path_prefix = "./data/"
    with open(path_prefix + file_path, "r") as data_file:
        if file_path.endswith(".json"):
            file_content = json.load(data_file)
        else:
            raise Exception(
                "Invalid filetype for file: " + file_path + ". Must end with .json"
            )
    return file_content
