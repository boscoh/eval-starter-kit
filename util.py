import yaml

MAX_LEN_LINE = 80


def block_str_yaml_representer(dumper, data):
    if isinstance(data, str) and ("\n" in data or len(data) > MAX_LEN_LINE):
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


yaml.add_representer(str, block_str_yaml_representer)


def load_yaml(file_path: str) -> dict:
    with open(file_path, "r") as f:
        return yaml.safe_load(f)


def save_yaml(data: dict, file_path: str):
    with open(file_path, "w") as f:
        yaml.dump(
            data,
            f,
            allow_unicode=True,
            default_flow_style=False,
            sort_keys=False,
            width=MAX_LEN_LINE,
            indent=2,
        )
