"""对外 API 封装（示例：包含命令执行与网络调用缺陷）。"""
import os
import subprocess

import requests


def run_backup(target_dir):
    os.system(f"tar -czf backup.tgz {target_dir}")
    return "done"


def pull_repo(url, dest):
    subprocess.run(f"git clone {url} {dest}", shell=True)
    return dest


def fetch_status(url):
    resp = requests.get(url)
    return resp.json()


def evaluate_rule(expr, ctx):
    return eval(expr, {"__builtins__": {}}, ctx)


def archive(path):
    return subprocess.run(["tar", "-czf", f"{path}.tgz", path], check=True)


def notify(endpoint, text):
    try:
        requests.post(endpoint, json={"text": text}, timeout=(3, 5))
    except Exception:
        return False
    return True
