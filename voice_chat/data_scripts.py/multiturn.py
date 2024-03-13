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

from typing import List, Dict, Any, Optional, Tuple
import re
import random

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


def timestamp_prefix():
    prefix = datetime.now().timestamp()
    return str(prefix)


def random_prefix(length: int = 6):
    prefix = str(uuid4())[:length]
    return prefix


def save_data(results: List[List[str]], menus: Dict) -> None:
    # save results to local data folder.
    if not os.path.exists("data"):
        os.mkdir("data")
    json_results = []
    timestamp = timestamp_prefix()

    filename = f"data/{timestamp}_data.json"
    json_results = [json.loads(datum) for datum in results]

    with open(filename, "w") as f:
        output = json.dumps(json_results, indent=4)
        f.write(output)

    filename = f"data/{timestamp}_menus.json"
    with open(filename, "w") as f:
        f.write(json.dumps(menus, indent=4))

    # Merge the files.
    for run in json_results:
        for result in run:
            result["menu"] = json.loads(menus[result["menu"].strip("#")])

    filename = f"data/{timestamp}_merged.json"
    with open(filename, "w") as f:
        f.write(json.dumps(json_results, indent=4))


REPEATS_PER_PROMPT = 5


# Prompt library used to generate synthetic prompts of multi-turn conversations.
rules = [
    """
         You do not make up answers about the menu if the item is not on the menu.
         If there are any options listed explicitly or implicitly in the menu item then you must let the customer know the options available and ask which one they would like. The options might be indicated by the word 'or' or listed in the 'other' section, or implicitly, by the appearance of more than one price option for a drink which corresponds to small and large size (e.g $4 / $5.5 means small size if $4 and $5.5 is large size)
         Are a polite but do not be too friendly or excited.
         """
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


def sample_menu(food_items: int = 5, beverage_categories: int = 4) -> Tuple[str, str]:
    food_idx = list(range(len(json_menu["food_items"])))
    random.shuffle(food_idx)
    food_idx = food_idx[:food_items]
    food_sample = [json_menu["food_items"][i] for i in food_idx[:food_items]]
    categories = json_menu.keys()
    categories = [
        category for category in categories if category != "food_items"
    ]  # remove the food options
    categories_idx = list(range(len(categories)))
    random.shuffle(categories_idx)
    categories_idx = categories_idx[:beverage_categories]
    categories_sample = {
        categories[_idx]: json_menu[categories[_idx]] for _idx in categories_idx
    }

    ret: Dict = {"food_items": food_sample}
    ret.update(categories_sample)

    menu_id = random_prefix(6)
    return menu_id, json.dumps(ret)


system_prompt = f"You are a casual waiter in a restaurant. Your job is to collect food and drink orders from customers and answer their questions. Your first task is to create a conversation dataset based on the FULL_MENU provided below in json format for plausible conversations with customers. ###rules### \r\n\r\n"

instructions = [
    f"""FULL_MENU : ###menu###\r\n\r\n
        Provide your answer in JSON format with the following schema, where I have included a description of the value for the given key in the answer:
        'menu': ###menu_id### a placeholder value that can be ignored, 
        'conversation': a list of less than turns of conversation between the customer and you, the waiter, where the customer ###ask### during the conversation. This value should be in json format where each turn in the conversation has 3 keys: 'customer' and 'waiter' which record the response from that person; and 'menu_action': a string from the list ["add","remove","none"] which indicates is in this turn of the conversation the customer is asking to add, remove an item otherwise its "none", 'menu_item': the menu item which the menu_action relates to if the menu_action is not "none" otherwise this is "none",
        'final_order': a list of the MENU item ordered, any variation the customer asked for and the price.

        Repeat this data generation task ###repeats### times and only return the final list of results.
        """
]

asks = [
    "asks to add one food item and one drink item from the menu value",
    "asks for one food item and then one drink but then then changes their mind and asks for a different drink.",
    "asks for food that is not on the menu (and you apologise), then the picks another item instead.",
    "order food and drink and ask to substitute an ingredient in the food.",
]


agent: Agent = Agent(
    prompt_driver=OpenAiChatPromptDriver(
        model="gpt-4-turbo-preview",
        api_key=os.environ["OPENAI_API_KEY"],
        temperature=0.6,
    )
)

results: List[str] = []
menus: Dict = {}


for instruction in instructions:
    for ask in asks:
        ## replace rules and asks
        menu_prefix, menu = sample_menu(5, 3)
        current_prompt = (
            system_prompt.replace("###rules###", rules[0])
            + "\r\n\r\n"
            + instruction.replace("###menu###", menu)
            .replace("###menu_id###", f"###{menu_prefix}###")
            .replace("###ask###", ask)
            .replace("###repeats###", str(REPEATS_PER_PROMPT))
        )
        response = agent.run(current_prompt).output.to_text()
        results.append(response)
        menus.update({menu_prefix: menu})

    save_data(results=results, menus=menus)
