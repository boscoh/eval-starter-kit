import yaml
import logging

logger = logging.getLogger(__name__)

MAX_LEN_LINE = 80


# Alternative: If you want to use folded style (>) instead of literal (|)
# This can be better for long paragraphs as it preserves line breaks only for paragraphs
def folded_str_yaml_representer(dumper, data):
    if isinstance(data, str):
        if "\n" in data or len(data) > 60:
            logger.info(f"Using literal style for multi-line string")
            # Use literal style for better readability of multi-line strings
            return dumper.represent_scalar("tag:yaml.org,2002:str", data, style=">")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style='' if "\n" not in data else "|")

yaml.add_representer(str, folded_str_yaml_representer)


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
            default_style=None,
            explicit_start=True,
        )

