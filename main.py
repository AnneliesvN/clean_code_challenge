import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
from datetime import *

supply_words = ["pan", "rasp", "kom"]


def clean_str(text: str):
    """Clean text by separating all the words and removing punctuation."""
    str_list = [''.join(x for x in txt if x.isalnum()) for txt in text.split()]
    str_list = [txt.lower() for txt in str_list]
    return str_list


def get_recipes():
    """Load and parse lunch recipes input."""
    df = pd.read_csv("data/lunch_recipes.csv")
    for wrd in supply_words:
        # count the amount of times a word occurs in the recipe.
        df[f"{wrd}"] = df.recipe.apply(
            lambda text: clean_str(text).count(wrd) > 0)
        df[f"{wrd}"] = df[f"{wrd}"].apply(lambda x: x is True)
    df = df.drop(columns=['servings', 'recipe', 'url', 'dish'])
    df['date'] = df.date.apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
    return df


def get_attendance():
    """Load and parse attendance input."""
    df = pd.read_csv("data/key_tag_logs.csv")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date
    df['time'] = df['timestamp'].dt.time

    # get attendance on each date and select for people who were present during lunch time
    df_attendance = df.pivot(index=['date', 'name'], columns=["event"], values="time")
    df_lunch_attendance = df_attendance[
        (df_attendance['check in'] < time(12, 0, 0)) &
        (df_attendance['check out'] > time(12, 0, 0))
        ].copy()

    # unpivot and clean up
    df_lunch_attendance['attendance'] = 1
    result = (
        df_lunch_attendance
        # get attendance per name
        .drop(columns=['check in', 'check out'])
        .unstack(level=1)
        .droplevel('event', axis=1)
        # attendance output is either 0 or 1
        .fillna(0)
        .astype(int)
        # get date as column and reset axis
        .reset_index()
        .rename_axis(None, axis=1)
    )
    return result


def get_dishwasher_log():
    """Load and parse dishwasher input."""
    df = pd.read_csv("data/dishwasher_log.csv")
    df["date"] = df.date.apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
    return df


def train_model():
    """Train prediction model."""
    df_recipes = get_recipes()
    df_attendance = get_attendance()
    df_dishwasher = get_dishwasher_log()

    df = df_recipes.merge(df_attendance,
                          on="date",
                          how="outer").merge(df_dishwasher).fillna(0)
    reg = LinearRegression(fit_intercept=False, positive=True) \
        .fit(df.drop(["dishwashers", "date"], axis=1),
             df["dishwashers"])
    return dict(zip(reg.feature_names_in_,
                    [round(c, 3) for c in reg.coef_]))


if __name__ == "__main__":
    print(train_model())
