"""
Create plausible multiturn conversations between waiter and customer
Scenarios to be encountered:
Add and remove items in casual language
Ask question about ingredient
Ask for an item with an ingredient modified (add, removed, replaced)
Ask for an item that isn't on the menu.
No drink has been ordered.
Ask for price.

Data format. Dataset will be used for other finetuning so additional items collected.
=====================================================================================

{'system_prompt':'...', prompt: '...','completion':...,'is_order':bool, current_order: [ ]}
system prompt should contain part of menu
prompt should be multi-turns seperated by [user], [assistant] tokens.
completion is the answer

"""

from voice_chat.configs import AppConfig

Configurations = AppConfig.Configurations
from voice_chat.utils import createFolderIfMissing, createIfMissing

from typing import List, Dict, Any, Optional, Tuple
import re
import random
import logging

from griptape.structures import Agent
from griptape.utils import Chat, PromptStack
from griptape.drivers import (
    HuggingFaceInferenceClientPromptDriver,
    OpenAiCompletionPromptDriver,
    OpenAiChatPromptDriver,
)
from griptape.artifacts import TextArtifact

from dotenv import load_dotenv
import os, json
from datetime import datetime
from uuid import uuid4

load_dotenv()  # get environmet variables as the openai prompt driver expects the credentials to be there.
logger = logging.getLogger("SynthDataLogger")
logger.setLevel(logging.DEBUG)
log_file_path = os.path.join(Configurations.logging.root_folder, "synth_data_logs.log")
createIfMissing(log_file_path)


def timestamp_prefix():
    prefix = datetime.now().timestamp()
    return str(prefix)


def random_prefix(length: int = 6):
    prefix = str(uuid4())[:length]
    return prefix


RUN_PREFIX = timestamp_prefix() + "_" + random_prefix(3)
logger.info(f"Run label : {RUN_PREFIX}")


def clear_markdown(string: str) -> str:
    ret = re.sub(r"^.*?\[", "[", string, flags=re.DOTALL)
    ret = re.sub(r"\][^]]*$", "]", ret, flags=re.DOTALL)
    return ret


def save_data(results: List[List[str]], menus: Dict, subfolder: str = "0") -> bool:
    """A resilient save function that saves conversations to local data folder.
    @return:
        bool: indicating if any data was successfully saved.
    """
    if not os.path.exists("data"):
        os.mkdir("data")
    json_results = []
    timestamp = timestamp_prefix()

    root_folder = f"data/{RUN_PREFIX}/{subfolder}/"
    createFolderIfMissing(root_folder)

    json_results = []
    try:
        json_results = [json.loads(clear_markdown(datum)) for datum in results]
    except Exception as e:
        logger.error(
            f"Error parsing entire results list. Iteratively drop run from results to remove corrupt Dialog jsons."
        )
        # try suceissive truncations
        # Adhoc generation errors.
        for _ in range(len(results)):
            results = results[:-1]
            if len(results) > 0:
                try:
                    json_results = [
                        json.loads(clear_markdown(datum)) for datum in results
                    ]
                    break
                except Exception as e:
                    logger.error("Dropping last run..")
            else:
                logger.error("Unable recover result by dropping results. Skipped")
                return False

    filename = f"{root_folder}{timestamp}_data.json"
    with open(filename, "w") as f:
        output = json.dumps(json_results, indent=2)
        f.write(output)

    filename = f"{root_folder}{timestamp}_menus.json"
    with open(filename, "w") as f:
        f.write(json.dumps(menus, indent=2))

    # Merge the files.
    for run in json_results:
        if not isinstance(run, list):
            run = [run]
        for result in run:
            result["menu"] = json.loads(menus[result["menu"].strip("#")])

    filename = f"{root_folder}{timestamp}_merged.json"
    with open(filename, "w") as f:
        f.write(json.dumps(json_results, indent=4))

    return True


# Prompt library used to generate synthetic prompts of multi-turn conversations.
rules = [
    """ You do not make up answers about the menu if the item is not on the menu.
         If there are any options listed explicitly or implicitly in the menu item then you must let the customer know the options available and ask which one they would like. The options might be indicated by the word 'or' or listed in the 'other' section, or implicitly, by the appearance of more than one price option for a drink which corresponds to small and large size (e.g $4 / $5.5 means small size if $4 and $5.5 is large size).
         If the customer hasn't added a drink to their order then you should ask if they'd like anything to drink. If they have only ordered a drink then ask if they would like any food.
         You are a polite but do not be too friendly or excited.""",
    """
You are a waiter in a casual restaurant. Your JOB is to navigate towards a specified GOAL. The goal is to sell a meal containing food and drinks from the following MENU included below and in json format.

In the process you NEED to evaluate the current conversation HISTORY and determine which stage the discussion is on. Based on the discussion, you need to answer the QUESTION and provide a follow-up question.

ALWAYS check the history and use as a reference while answering the QUESTION.

Your GOAL path is as below:

1. If the customer greets you, you greet them back and ask what what they would like to have today.
2. If a customer makes a statement instead of asking a question then their statement is an answer to the last question you asked. 
3. If a customer asks about a menu item, answer their question briefly and ask if they would like it.
4. If a custom orders an menu item and that menu item has some options, then you must tell them the options and ask which option they’d like to order. 
5. If a menu item is a beverage and that item has multiple prices listed in the menu then you can assume that the prices relate to different size options of small and large.
6. If a customer has only ordered food then you must ask if they would like a drink with that. 
7. If a customer has only ordered a drink they you must ask if they would like any food with that. 

Always read the customers order back to them at the end, including and changes they made, and tell them the total price.
""",
]
menu = """
{
    "food_items": [
        {
            "name": "RUSTIC TOAST",
            "price": "$8.50",
            "descriptions": [
                "Sourdough or Ciabatta",
                "Fig & Lemon Jam",
                "Orange Marmalade",
                "Strawberry Jam",
                "Peanut Butter or Vegemite"
            ]
        },
        {
            "name": "HOT CHEESE",
            "price": "$9.50",
            "descriptions": [
                "With Tomato & Oregano on Sourdough"
            ]
        },
        {
            "name": "FRUIT TOAST",
            "price": "$8.50",
            "descriptions": [
                "Apricots, Dates, Figs & Nuts, Toasted with Butter"
            ]
        },
        {
            "name": "GRANOLA CUP GF",
            "price": "$10.50",
            "descriptions": [
                "House-made Granola",
                "Greek Yogurt",
                "Strawberry Jam and Strawberries"
            ]
        },
        {
            "name": "ACAI BOWL GF",
            "price": "$17.50",
            "descriptions": [
                "House-made Granola",
                "Banana",
                "Strawberry"
            ],
            "other": [
                "Add Peanut Butter - $2"
            ]
        },
        {
            "name": "ITALIAN OMELETTE",
            "price": "$23",
            "descriptions": [
                "Tomato",
                "Mozzarella",
                "Topped with Basil & Walnut Pesto",
                "Mixed Leaves",
                "Ciabatta"
            ]
        },
        {
            "name": "AVO BRUSCHETTA",
            "price": "$22",
            "descriptions": [
                "On Sourdough",
                "Heirloom Tomatoes",
                "Feta",
                "Secret Salt",
                "Basil & Walnut Pesto"
            ],
            "other": [
                "Add Poached Egg - $3"
            ]
        },
        {
            "name": "SOUP OF THE DAY",
            "price": "$16",
            "descriptions": [
                "Rustic Bread"
            ]
        },
        {
            "name": "TWIST BIG BREAKFAST",
            "price": "$28",
            "descriptions": [
                "Two Poached or Fried Eggs",
                "Gourmet Pork Sausages",
                "Haloumi",
                "Grilled Tomato",
                "Spinach",
                "Sourdough"
            ]
        },
        {
            "name": "BREKKY ROLL",
            "price": "$13",
            "descriptions": [
                "Bacon & Egg on a Milk Bun"
            ],
            "other": [
                "Sauce: Tomato, BBQ, Chipotle Mayo or Aioli",
                "Add Cheese - $1, Smashed Avo - $2"
            ]
        },
        {
            "name": "TWIST WINTER BUN",
            "price": "$17",
            "descriptions": [
                "Bacon or Haloumi",
                "Egg",
                "Tomato",
                "Caramelized Onion",
                "Mixed Leaves",
                "Melted Cheese",
                "Aioli"
            ],
            "other": [
                "Add Smashed Avo - $2"
            ]
        },
        {
            "name": "MEXICAN eggs",
            "price": "$23",
            "descriptions": [
                "Mixed Beans Cooked with Chili, Garlic, Onion and Herbs in a Rich Tomato Sauce",
                "Baked with Mozzarella",
                "Two Poached Eggs",
                "Ciabatta"
            ]
        },
        {
            "name": "EGGS BENEDICT",
            "price": "$26",
            "descriptions": [
                "Two Poached Eggs",
                "Spinach",
                "Hollandaise Sauce",
                "Served on Sourdough"
            ],
            "other": [
                "Choice of: Salmon, Bacon or Ham"
            ]
        },
        {
            "name": "EGGS POACHED OR FRIED",
            "price": "$13",
            "descriptions": [
                "Served on Sourdough or Ciabatta"
            ],
            "other": [
                "Add Cheese - $1, Smashed Avo - $2"
            ]
        }
    ],
    "extra_items": {
        "Salmon, Gourmet Pork Sausages": "$6",
        "Bacon, Haloumi": "$5",
        "Avocado, Grilled Tomato": "$4",
        "Free-Range Egg": "$3",
        "Gluten-Free Bread": "$2"
    },
    "hot_drinks": {
            "CAPPUCCINO, LATTE, FLAT WHITE": "$4 / $4.8",
            "PICCOLO, MACCHIATO": "$4.2",
            "ESPRESSO": "$3.8",
            "MOCHA": "$5, $6",
            "HOT CHOCOLATE": "$4.5 / $5.5",
            "CHAI LATTE": "$4.5",
            "MASALA CHAI TEA Brewed with milk": "$6",
            "PRANA CHAI TEA Brewed with milk": "$6",
            "TURMERIC LATTE": "$5.50 / $6",
            "Soy, Oat": "$0.8",
            "Almond": "$1",
            "Extra Shot": "$0.5",
            "Decaf": "$0.7"
    },
    "iced_drinks": {
        "ICED LONG BLACK": "$5.5",
        "ICED LATTE": "$6.2",
        "ICED COFFEE": "$10.5",
        "ICED CHOCOLATE": "$9",
        "AFFOGATO": "$7"
    },
    "tea": {
        "T-NOMICS ORGANIC LEAF TEA": "$5",
        "Varieties": [
            "English Breakfast",
            "Earl Grey",
            "Peppermint",
            "Lemongrass & Ginger",
            "Golden Oolong Green"
        ]
    },
    "cold_drinks": {
        "PUREZZA PREMIUM WATER, Bottomless Sparkling": "$8",
        "Juices": [
            "ORANGE JUICE - $7.5",
            "APPLE JUICE - $7.0"
        ],
        "Sodas": [
            "Watermelon & Mint - $6",
            "Passionfruit Lemonade - $6"
        ],
        "Milkshakes": [
            "Chocolate - $10",
            "Vanilla - $12",
            "Caramel - $10",
            "Strawberry - $12"
        ],
        "Milk Alternatives": {
            "Soy, Almond, Oat": "$1"
        }
    },
    "smoothies": {
        "price":"$11",
        "Flavors": [
            "Banana",
            "Mixed Berry",
            "Acai"
        ],
        "Milk Alternatives": {
            "Soy, Almond, Oat": "$1"
        }
    },
    "protein_shake": { 
        "price":"$14",
        "Ingredients": [
            "Peanut butter",
            "Modern pea protein",
            "Almond milk",
            "Banana",
            "Honey & ice"
        ],
        "Additions": [
            "Add shot of coffee $0.5"
        ]
    },
    "immune_booster_shake": {
        "price":"$14",
        "Ingredients": [
            "Green apple",
            "Banana",
            "Spinach",
            "Mint",
            "Lime",
            "Ginger"
        ]
    },
    "cocktails": [
        {
            "name": "APEROL SPRITZ",
            "price": "$16",
            "ingredients": [
                "Aperol",
                "Prosecco",
                "Soda"
            ]
        },
        {
            "name": "MARGARITA",
            "price": "$18",
            "ingredients": [
                "Tequila",
                "Citrus",
                "Orange liqueur"
            ]
        },
        {
            "name": "ESPRESSO MARTINI",
            "price": "$18",
            "ingredients": [
                "Coffee",
                "Vodka",
                "Kahlua"
            ]
        }
    ],
    "beer": [
        {
            "name": "BUDVAR / CORONA",
            "price": "$9"
        },
        {
            "name": "PERONI",
            "price": "$8"
        }
    ]
}

"""

json_menu: Dict = json.loads(menu)


def sample_menu(
    food_item_count: int = 4,
    non_food_group_count: int = 4,
    include_extras: bool = True,
    extras_key: str = "extra_items",
) -> Tuple[str, str]:
    """Down sample the menu for inclusion in the prompt"""

    food_idx = list(range(len(json_menu["food_items"])))
    random.shuffle(food_idx)
    food_idx = food_idx[:food_item_count]
    food_sample = [json_menu["food_items"][i] for i in food_idx[:food_item_count]]
    all_items = json_menu.keys()
    all_items = [
        category for category in all_items if category != "food_items"
    ]  # remove the food options. food_item is assumed top level in schema.
    non_food_groups_idx = list(range(len(all_items)))
    random.shuffle(non_food_groups_idx)
    non_food_groups_idx = non_food_groups_idx[:non_food_group_count]
    non_food_samples = {
        all_items[_idx]: json_menu[all_items[_idx]] for _idx in non_food_groups_idx
    }
    if not "extra_items" in non_food_samples:
        non_food_samples.update({"extra_items": json_menu["extra_items"]})

    ret: Dict = {"food_items": food_sample}
    ret.update(non_food_samples)

    menu_id = random_prefix(6)
    return menu_id, json.dumps(ret)


system_prompts = [
    f"You are a casual waiter in a restaurant. Your job is to collect food and drink orders from customers and answer their questions. Your first task is to create a dialog dataset based on the FULL_MENU provided below, in json format, for plausible conversations with customers. ###rules### \r\n",
    f"You are a synthetic data generator for multiturn Dialogs between a waiter in a restaurant and a customer. Your task is to generate realistic Dialogs between a waiter and a customer while the customer is making their order, enquiring about menu items or asking for modifications to menu items available in the FULL_MENU included below. ###rules###\r\n",
    f"Your task is to create a dialog dataset for conversations between a waiter and a customer when ordering from the FULL_MENU provided below, in json format. You must generate plausible conversations with customers by following these RULES.\n RULES:###rules### \r\n",
]

system_prompt = system_prompts[2]

instruction_library = [
    """FULL_MENU : ###menu###\r\n
        Generate a list of ###repeats### Conversations where each Conversation is a JSON string with the schema below. Each key includes instructions for generating the value for the given key:
        'menu': ###menu_id### a placeholder value that can be ignored, 
        'conversation': a list of less than 5 conversation turns between the customer and you, the waiter, where the customer ###ask### during the conversation. This value should be in json format where each turn in the conversation has 3 keys: 'customer' and 'waiter' which record the response from that person; and 'menu_actions': a list of dictionaries, each with the following 3 keys: 'action' a string from the list ["add","remove","update, "none"] where 'add' indicates the customer has asked to add menu item, not an ingredient of a menu item, to the order and even if they have not finalized the choices for that menu item,'remove' means the customer has asked to remove a menu item (not an ingredient) and 'update' means the customer has selected an option for menu item or requested an ingredient of a menu item to be varied, otherwise 'none', 'menu_item': the menu item which the menu_action relates when menu_item is not none, otherwise set this to none; and 'variation': the requested change and the ingredient, size or special request from the customer when menu_action value update.
        'final_order': a list of dictionaries with 3 keys: 'name' the name of the menu item ordered, 'variation': variations to that menu item, if any, 'price' the total price for that menu item including any charges for the variations or special requests for that meny item.
        """,
    """FULL_MENU : ###menu###\r\n
        In this Dialog the customer ###ask###. Each complete Dialog will be recorded in JSON format using the schema described below and between the <schema> ... </schema> tags, where the description in the value for a key contains instructions for generating that value. You must generate ###repeats### Dialogs. Return the dialogs as a list of strings using the format [Dialog1,Dialog2,...]. This is not markup or markdown so the first character returned should be the opening quare bracket of the list.
        <schema>
        'menu': ###menu_id### a placeholder value that can be ignored, 
        'conversation': a list of less than 5 conversation turns between the customer and you, the waiter, where the customer ###ask### during the conversation. This value should be in json format where each turn in the conversation has 3 keys: 'customer' and 'waiter' which record the response from that person; and 'menu_actions': a list of dictionaries, each with the following 3 keys: 'action' a string from the list ["add","remove","update, "none"] where 'add' indicates the customer has asked to add menu item, not an ingredient of a menu item, to the order and even if they have not finalized the choices for that menu item,'remove' means the customer has asked to remove a menu item (not an ingredient) and 'update' means the customer has selected an option for menu item or requested an ingredient of a menu item to be varied, otherwise 'none', 'menu_item': the menu item which the menu_action relates when menu_item is not none, otherwise set this to none; and 'variation': the requested change and the ingredient, size or special request from the customer when menu_action value update.
        'final_order': a list of dictionaries with 3 keys: 'name' the name of the menu item ordered, 'variation': variations to that menu item, if any, 'price' the total price for that menu item including any charges for the variations or special requests for that meny item.
        </schema>
        """,
    """FULL_MENU : ###menu###\r\n
        In this Dialog the customer ###ask###. Each complete Dialog will be recorded in JSON format using the schema described below and between the <schema> ... </schema> tags, where the description in the value for a key contains instructions for generating that value. You must generate ###repeats### Dialogs. Return the dialogs as a list of strings using the format [Dialog1,Dialog2,...]. This is not markup or markdown so the first character returned should be the opening quare bracket of the list.
        <schema>
        'menu': ###menu_id### a placeholder value that can be ignored, 
        'conversation': a list of less than 10 conversation turns between the customer and you, the waiter, where the customer ###ask### during the conversation. This value should be in json format where each turn in the conversation has 3 keys: 'customer' and 'waiter' which record the response from that person; and 'menu_actions': a list of dictionaries, each with the following 3 keys: 'action' a string from the list ["add","remove","update, "none"] where 'add' indicates the customer has asked to add menu item, not an ingredient of a menu item, to the order and even if they have not finalized the choices for that menu item,'remove' means the customer has asked to remove a menu item (not an ingredient) and 'update' means the customer has selected an option for menu item or requested an ingredient of a menu item to be varied, otherwise 'none', 'menu_item': the menu item which the menu_action relates when menu_item is not none, otherwise set this to none; and 'variation': the requested change and the ingredient, size or special request from the customer when menu_action value update.
        </schema>
        """,
]

instructions = [instruction_library[2]]

asks = [
    "asks for one food item",
    "equires about one menu item but asks for a different one",
    "asks about the ingredients in one menu item and then places an order for that item",
    "asks for one food item and then one drink item from the menu",
    "asks for one food item and then one drink, then then changes their mind and asks for a different drink",
    "asks for food that is not on the menu (and you apologise), then the picks another item instead",
    "order food and drink and ask to substitute an ingredient in the food",
]
""" asks = ["orders a food item, then changes their mind and chooses another food item, then finally asks for an ingredient sustitution."] """


agent: Agent = Agent(
    prompt_driver=OpenAiChatPromptDriver(
        model="gpt-4-0125-preview",
        api_key=os.environ["OPENAI_API_KEY"],
        temperature=0.6,
    )
)

results: List[str] = []
menus: Dict = {}

REPEATS_PER_PROMPT = 4

for instruction_idx in range(len(instructions)):
    for ask_id in range(len(asks)):
        ## replace rules and asks
        menu_prefix, menu = sample_menu(4, 3)
        current_prompt = (
            system_prompt.replace("###rules###", rules[1])
            + "\r\n\r\n"
            + instructions[instruction_idx]
            .replace("###menu###", menu)
            .replace("###menu_id###", f"###{menu_prefix}###")
            .replace("###ask###", asks[ask_id])
            .replace("###repeats###", str(REPEATS_PER_PROMPT))
        )
        response = agent.run(current_prompt).output.to_text()
        results.append(response)
        menus.update({menu_prefix: menu})

        if len(results) > 4:
            save_data(results=results, menus=menus, subfolder=str(instruction_idx))
            results = []
            menus = {}

    save_data(results=results, menus=menus, subfolder=str(instruction_idx))
    results = []
    menus = {}
