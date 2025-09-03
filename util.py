import logging

import yaml
from path import Path

logger = logging.getLogger(__name__)

MAX_LEN_LINE = 80


def folded_str_yaml_representer(dumper, data):
    if isinstance(data, str):
        if "\n" in data or len(data) > 60:
            return dumper.represent_scalar("tag:yaml.org,2002:str", data, style=">")
    return dumper.represent_scalar(
        "tag:yaml.org,2002:str", data, style="" if "\n" not in data else "|"
    )


yaml.add_representer(str, folded_str_yaml_representer)


def load_yaml(file_path: str) -> dict:
    with open(file_path, "r") as f:
        return yaml.safe_load(f)


def save_yaml(data: dict, file_path: str):
    parent = Path(file_path).parent
    if parent:
        parent.makedirs_p()
    with open(file_path, "w") as f:
        yaml.dump(
            data,
            f,
            allow_unicode=True,
            default_flow_style=False,
            sort_keys=False,
            width=MAX_LEN_LINE,
            indent=2,
            default_style=None,
            explicit_start=True,
        )


def write_text(text: str, file_path: str):
    file_path = Path(file_path)
    if file_path.parent:
        file_path.parent.makedirs_p()
    file_path.write_text(text)
