"""
Takes in all the menu items sampled from the Kaggle Dataset "Uber Eats USA Restaurant and Menus
https://www.kaggle.com/datasets/ahmedshahriarsakib/uber-eats-usa-restaurants-menus
63k+ USA restaurants and 5 million+ menus from Uber Eats
"""

import pandas as pd
import json
from datetime import datetime
import random
import string
import os

data_path = (
    "/home/mtman/Documents/Repos/yakwith.ai/data/cafes_menu_long_descriptions_v2.csv"
)
home_path = "/home/mtman/Documents/Repos/yakwith.ai/data"
first_row_is_titles = True
MENU_LENGTH = 15

data = pd.read_csv(data_path, header=0, index_col=False)
nested_data = data.groupby("restaurant_id")
long_menus = nested_data.filter(lambda x: len(x) > MENU_LENGTH)
business_count = long_menus["restaurant_id"].unique()
print(
    f"Number of businesses with menus longer than {MENU_LENGTH} = {len(business_count)}"
)
long_menus = long_menus.groupby("restaurant_id")


def random_string(length=6):
    """Generate a random string of ASCII characters."""
    return "".join(
        random.choice(string.ascii_letters + string.digits) for _ in range(length)
    )


def order_prefix(length=3):
    return os.path.join(
        home_path, "".join([str(datetime.now().timestamp()), random_string(3)])
    )


menu_list = []


def get_menu_items(menu_items: pd.DataFrame) -> pd.DataFrame:
    menu_items.drop(columns=["restaurant_id"], inplace=True)
    ret = {
        "menu": json.loads(
            menu_items.to_json(orient="records", force_ascii=True, index=False)
        )
    }
    return ret


menu_list = [get_menu_items(menu_items) for restaurant_id, menu_items in long_menus]

filename = order_prefix(3) + ".csv"
print(f"Export to {filename}")
with open(filename, "w", newline="") as fp:
    for item in menu_list:
        fp.write(json.dumps(item, indent=2) + "\t")
