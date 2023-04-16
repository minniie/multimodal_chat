import os
import json
from argparse import ArgumentTypeError


def safe_mkdir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def load_json(dir_path, file_name):
    if not file_name.endswith(".json"):
        raise ArgumentTypeError(f"{file_name} should have .json extension.")
    file_path = os.path.join(dir_path, file_name)
    if os.path.exists(file_path):
        with open(file_path, "r", encoding='utf-8') as f:
            return json.load(f)
    else:
        return []


def dump_json(data, dir_path, file_name):
    if not file_name.endswith(".json"):
        raise ArgumentTypeError(f"{file_name} should have .json extension.")
    file_path = os.path.join(dir_path, file_name)
    with open(file_path, "w", encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def save_dialog(dialog, dir_path, user, date, file_name):
    user_path = os.path.join(dir_path, user, date)
    safe_mkdir(user_path)
    dialogs = load_json(user_path, file_name)
    dialogs.append(dialog)
    dump_json(dialogs, user_path, file_name)
    workload = len(dialogs)
   
    return workload