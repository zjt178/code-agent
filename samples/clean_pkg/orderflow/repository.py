"""订单仓储（规范实现：上下文管理器、参数化查询、具体异常）。"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path


class OrderRepository:
    """基于 SQLite 的订单读写。"""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path

    def connect(self) -> sqlite3.Connection:
        """打开连接（调用方负责关闭）。"""
        return sqlite3.connect(self.db_path)

    def get(self, conn: sqlite3.Connection, order_id: int) -> dict | None:
        """按 id 查订单，不存在返回 None。"""
        cursor = conn.execute("SELECT payload FROM orders WHERE id = ?", (order_id,))
        row = cursor.fetchone()
        return json.loads(row[0]) if row else None

    def save(self, conn: sqlite3.Connection, order_id: int, payload: dict) -> None:
        """写入或覆盖订单。"""
        conn.execute(
            "INSERT INTO orders (id, payload) VALUES (?, ?) "
            "ON CONFLICT(id) DO UPDATE SET payload = excluded.payload",
            (order_id, json.dumps(payload, ensure_ascii=False)),
        )
        conn.commit()


def read_manifest(path: str) -> dict:
    """读取清单文件；文件缺失时抛 FileNotFoundError 由调用方处理。"""
    file = Path(path)
    with file.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def load_orders(path: str) -> list[dict]:
    """批量读取订单 json 数组。"""
    data = read_manifest(path)
    if not isinstance(data, list):
        raise TypeError("清单格式应为数组")
    return data
