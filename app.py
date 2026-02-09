import streamlit as st
import pandas as pd
import numpy as np
import math

def entropy(col):
    values, counts = np.unique(col, return_counts=True)
    return -sum((c / len(col)) * math.log2(c / len(col)) for c in counts)

def info_gain(df, attr, target):
    total_entropy = entropy(df[target])
    vals = df[attr].unique()
    weighted_entropy = sum(
        (len(df[df[attr] == v]) / len(df)) * entropy(df[df[attr] == v][target])
        for v in vals
    )
    return total_entropy - weighted_entropy

def id3(df, target, attrs):
    if len(df[target].unique()) == 1:
        return df[target].iloc[0]
    if not attrs:
        return df[target].mode()[0]
    best = max(attrs, key=lambda a: info_gain(df, a, target))
    tree = {best: {}}
    for val in df[best].unique():
        sub_df = df[df[best] == val]
        tree[best][val] = id3(sub_df, target, [a for a in attrs if a != best])
    return tree
