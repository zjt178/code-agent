"""通用小工具（示例：包含可维护性缺陷）。"""
import functools
import itertools


def label_map(code):
    mapping = {"a": 1, "b": 2, "a": 3}
    return mapping.get(code, 0)


def build_flags(flags={}):
    flags["built"] = True
    return flags


def parse_kv(raw):
    result = {}
    for pair in raw.split("&"):
        if "=" in pair:
            key, value = pair.split("=", 1)
            if key:
                if value:
                    if len(value) > 1:
                        if value[0] == "1":
                            result[key] = True
                        else:
                            result[key] = value
    return result


def flatten(nested):
    out = []
    for item in nested:
        if isinstance(item, list):
            for sub in item:
                out.append(sub)
        else:
            out.append(item)
    return out


def with_cache(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)

    return wrapper


def pairs(items):
    return list(itertools.combinations(items, 2))
