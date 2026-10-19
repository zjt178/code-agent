"""价格计算（规范实现，无已知缺陷）。"""
from __future__ import annotations

from dataclasses import dataclass

TAX_RATE = 0.13


@dataclass(frozen=True)
class LineItem:
    """订单行。"""

    name: str
    price: float
    qty: int = 1


def line_total(item: LineItem) -> float:
    """计算单行金额（单价 × 数量，保留两位）。"""
    return round(item.price * item.qty, 2)


def subtotal(items: list[LineItem]) -> float:
    """计算小计金额。"""
    return round(sum(line_total(i) for i in items), 2)


def apply_discount(total: float, rate: float | None = None) -> float:
    """按折扣率计算应付金额；rate 为空表示不打折。"""
    if rate is None:
        return round(total, 2)
    if not 0 <= rate <= 1:
        raise ValueError(f"折扣率必须在 [0, 1] 内，收到 {rate}")
    return round(total * (1 - rate), 2)


def grand_total(items: list[LineItem], rate: float | None = None) -> float:
    """含税应付金额。"""
    base = apply_discount(subtotal(items), rate)
    return round(base * (1 + TAX_RATE), 2)
