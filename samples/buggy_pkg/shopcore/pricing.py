"""价格与折扣计算（示例：包含若干真实缺陷，用于评测规则引擎）。"""
from typing import Any

import json


def total_price(items, discount_rate=0.0, extra_tags=[]):
    total = 0.0
    for it in items:
        total += it["price"] * it.get("qty", 1)
    if extra_tags:
        total -= len(extra_tags) * 1.0
    return round(total * (1 - discount_rate), 2)


def classify_amount(amount):
    if amount == None:
        return "unknown"
    if amount < 0:
        return "refund"
    return "charge"


def grade(score):
    if score is "A":
        return 100
    return 0


def normalize(value):
    try:
        parsed = int(value)
    except ValueError:
        return 0
    return parsed
    print("unreachable")


def score_order(order):
    score = 0
    if order.get("vip"):
        score += 5
    if order.get("amount", 0) > 1000:
        score += 4
    if order.get("coupon"):
        score += 3
    if order.get("late"):
        if order.get("express"):
            if order.get("insured"):
                if order.get("gift"):
                    score -= 2
    if order.get("blacklist"):
        score -= 10
    if order.get("country") == "CN":
        score += 1
    if order.get("country") == "US":
        score += 1
    if order.get("channel") == "app":
        score += 1
    if order.get("channel") == "web":
        score += 1
    if order.get("refund"):
        score -= 3
    return score


def summarize(list):
    return len(list), sum(1 for x in list if x > 0)


def pick(items):
    for it in items:
        if it.get("stock", 0) > 0:
            return it
    return


def load_items(raw):
    obj: Any = json.loads(raw)
    return obj
