"""Create a dataset for making the function call that maintains the current order state.

The dataset will consist of
prompt
input: all customer/waiter turns prior to the current turn
the customer turn for the current turn
target: the order action

"""

from voice_chat.configs import AppConfig
from voice_chat.utils import createIfMissing

from typing import List, Dict
import csv
import logging

from dotenv import load_dotenv
import os
import json
from datetime import datetime
from uuid import uuid4

ROOT_PATH = "/home/mtman/Documents/Repos/yakwith.ai/data/1711405462.204613_881"

load_dotenv()  # get environmet variables as the openai prompt driver expects the credentials to be there.
Configurations = AppConfig.Configurations
logger = logging.getLogger("OrderFuncCallDataLogger")
logger.setLevel(logging.DEBUG)
log_file_path = os.path.join(
    Configurations.logging.root_folder, "order_func_calls_data_logs.log"
)
createIfMissing(log_file_path)


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
                     'conversation': [ {waiter: ..., customer:..., actions: ...}]}, ...]
                Note the customer always speaks first.
    """

    rules = [""" """]

    SYSTEM_PROMPT = [
        """You are a clever analyst. Your GOAL is to generate the parameters required by a function that keeps track of the current state of the ORDER at a cafe.
                      You are given a conversation between a waiter and a customer and you will generate parameters need to generate three parameters 
                      that determine how the current order will be changed by the last thing the customer says. This may involve reviewing the conversation
                      history to understand the context for the customers statement. 
                      Example 1.
                    <|im_start|>user
                    Can I get the steak and chips thanks<|im_end|>
                    <|im_start|>analyst
                    {'item':'steak with chips','action':'add','variation':''}<|im_end|>

                    Example 2.
                    <|im_start|>user
                    Can I get the steak and chips thanks<|im_end|>
                    <|im_start|>assistant
                    Sure. How would you like that cooked?<|im_end|>
                    <|im_start|>user
                    Medium rare thanks<|im_end|>
                    <|im_start|>analyst
                    {'item':'steak and chips', 'action':'update','variation':'medium rare'}<|im_end|>

                    Example 3.
                    <|im_start|>user
                    Can I get the steak and chips thanks<|im_end|>
                    <|im_start|>assistant
                    Sure. How would you like that cooked?<|im_end|>
                    <|im_start|>user
                    Medium rare thanks<|im_end|>
                    <|im_start|>assistant
                    And what sauce would you like on that?<|im_end|>
                    <|im_start|>user
                    Just tomato will be fine<|im_end|>
                    <|im_start|>analyst
                    {'item':'steak and chips','action':'update', 'variation':'add tomoato sauce'}<|im_end|>

                    Example 4.
                    <|im_start|>user
                    Can I get the steak and chips thanks<|im_end|>
                    <|im_start|>assistant
                    Sure. How would you like that cooked?<|im_end|>
                    <|im_start|>user
                    Medium rare thanks<|im_end|>
                    <|im_start|>assistant
                    And what sauce would you like on that?<|im_end|>
                    <|im_start|>user
                    Just tomato will be fine<|im_end|>
                    <|im_start|>assistant
                    Would you like anything to drink with that<|im_end|>
                    <|im_start|>user
                    A strawberry milk shake thanks<|im_end|>
                    <|im_start|>analyst
                    {'item':'milk shake','action':'add', variation:'strawberry'}<|im_end|>
                    """,
        """You are a clever conversation analyst. You are given a conversation between a waiter (the 'assistant') and a customer (the 'user') at a restaurant. Your TASK is to generate three parameters required by a FUNCTION that keeps track of the current state of the customer's ORDER. To understand the context of the customers final request, you will need to review the conversation history. The parameters must be returned in JSON format using the following schema:
                     {'item': the name of the current menu_item being discussed,
                      'action': how the ORDER should be changed: 'add','remove','update',
                      'variation': the choice or modification that the customer has asked for. If they have not made any choices then this value should be 'none'
                      }""",
    ]

    SYSTEM_PROMPT_CHOICE = 1

    final_list = []
    for dialog in dia_list:
        """system_prompt = SYSTEM_PROMPT.replace("###rules###", rules[0]).replace(
        "###menu###", json.dumps(dialog["menu"])"""

        system_prompt = SYSTEM_PROMPT[SYSTEM_PROMPT_CHOICE].replace(
            "###rules###", rules[0]
        )
        system_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n"

        ordered_conversations = [
            [
                {"customer": turn["customer"]},
                {"menu_actions": turn["menu_actions"]},
                {"waiter": turn["waiter"]},
            ]
            for turn in dialog["conversation"]
        ]
        flatten_list: List[List[str]] = []
        for row in ordered_conversations:
            flatten_list.extend(row)
        turn_accumulator = []  # accumulate everything except actions.
        for turn in flatten_list:
            # get a list of the progresing conversations ended with the action.
            if "customer" in turn:
                turn_accumulator.append(
                    f'<|im_start|>user\n{turn["customer"]}<|im_end|>\n'
                )
            if "waiter" in turn:
                turn_accumulator.append(
                    f'<|im_start|>assistant\n{turn["waiter"]}<|im_end|>\n'
                )
            if "menu_actions" in turn:
                current = [system_prompt]
                current.extend(turn_accumulator)
                current.append(
                    f'<|im_start|>analyst\n{turn["menu_actions"]}<|im_end|>\n'
                )
                current_string = "".join(current)
                final_list.append(current_string)

    with open(os.path.join(ROOT_PATH, "function_call_data.tsv"), "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        for string in final_list:
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
                                if key in ["customer", "waiter", "menu_actions"]:
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
