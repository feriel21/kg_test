import yaml
import os

def load_config(path="config/pipeline_config.yml"):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    dataset = cfg.get("dataset", "small_corpus")

    # Expand {dataset} placeholders
    expanded = {}
    for key, val in cfg["paths"].items():
        expanded[key] = val.replace("{dataset}", dataset)

    cfg["paths_expanded"] = expanded
    return cfg
