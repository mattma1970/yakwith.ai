"""Create a completion dataset for supervised fine tuning

Format of the dataset
<|im_start|>system
prompt <|im_end|>

"""

from configs import AppConfig
from utils import createIfMissing

from typing import List, Dict
import csv
import logging

from dotenv import load_dotenv
import os
import json
from datetime import datetime
from uuid import uuid4

Configurations = AppConfig.Configurations

load_dotenv()  # get environmet variables as the openai prompt driver expects the credentials to be there.
logger = logging.getLogger("CompletionDataLogger")
logger.setLevel(logging.DEBUG)
log_file_path = os.path.join(
    Configurations.logging.root_folder, "completion_data_logs.log"
)
createIfMissing(log_file_path)

ROOT_PATH = "/home/mtman/Documents/Repos/yakwith.ai/data/1711008943.631961_450"


def timestamp_prefix():
    prefix = datetime.now().timestamp()
    return str(prefix)


def random_prefix(length: int = 6):
    prefix = str(uuid4())[:length]
    return prefix


def find_merged_files(start_path):
    merged_files = []

    for root, dirs, files in os.walk(start_path):
        for file in files:
            if "_merged" in file:
                full_path = os.path.join(root, file)
                merged_files.append(full_path)

    return merged_files


def flat_chat_format(dia_list: List[Dict]) -> List[str]:
    """
    dia_list: List[{'menu': ...,
                     'conversation': [ {waiter: ..., customer:...}]}, ...]
                Note the customer always speaks first.
    """

    rules = [
        """ You do not make up facts about items on the menu.
            Where a menu item has choices and alternative ingredients you explore those with the customer before completing the order.
            You always read the order back to the customer at the end of the conversation and tell them the full price.
        """
    ]

    SYSTEM_PROMPT = "You are a casual waiter in a restaurant. Your job is to collect food and drink orders from customers and answer their questions. You follow these RULES: ###rules###. The MENU is as follows: ###menu### "

    string_list = []
    for dialog in dia_list:
        system_prompt = SYSTEM_PROMPT.replace("###rules###", rules[0]).replace(
            "###menu###", json.dumps(dialog["menu"])
        )
        temp = [f"<|im_start|>system\n{system_prompt}<|im_end|>\n"]
        for turn in dialog["conversation"]:
            temp.append(f'<|im_start|>user\n{turn["customer"]}<|im_end|>\n')
            temp.append(f'<|im_start|>assistant\n{turn["waiter"]}<|im_end|>\n')
        string_list.append("".join(temp))

    with open(os.path.join(ROOT_PATH, "completion_data.tsv"), "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        for string in string_list:
            writer.writerow([string])


def main():
    files = find_merged_files(ROOT_PATH)
    all_data = []
    for file in files:
        try:
            with open(file, "r") as f:
                _dialogs = f.read()
                dialog_batches = json.loads(_dialogs)

                # flatten and filter
                for dialogs in dialog_batches:
                    for dialog in dialogs:
                        _menu = dialog["menu"]
                        _conversation = dialog["conversation"]
                        _filtered_conversation = []
                        for turn in _conversation:
                            _turn = {}
                            for key, value in turn.items():
                                if key in ["customer", "waiter"]:
                                    _turn[key] = value
                            _filtered_conversation.append(_turn)

                        all_data.append(
                            {"menu": _menu, "conversation": _filtered_conversation}
                        )
        except Exception as e:
            logger.error(f"Error reading {e}")

    flat_chat_format(all_data)


if __name__ == "__main__":
    main()
