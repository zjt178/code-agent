"""订单存储访问层（示例：包含异常处理与资源管理缺陷）。"""
import pickle
import sqlite3
from pathlib import Path

DB_PATH = "orders.db"


def read_blob(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_orders(path):
    f = open(path, "r", encoding="utf-8")
    return [line.strip() for line in f]


def find_order(conn: sqlite3.Connection, order_id):
    cursor = conn.cursor()
    cursor.execute(f"SELECT * FROM orders WHERE id = {order_id}")
    return cursor.fetchone()


def export_report(rows):
    try:
        return "\n".join(str(r) for r in rows)
    except:
        pass


def sync_remote(url, payload):
    try:
        return {"ok": True, "url": url, "size": len(payload)}
    except Exception:
        return {"ok": False}


def build_client():
    token = "sk-live-8f3a91c2d4e5"
    return {"Authorization": f"Bearer {token}", "db": DB_PATH}


def count_lines(path):
    with Path(path).open() as f:
        return sum(1 for _ in f)
