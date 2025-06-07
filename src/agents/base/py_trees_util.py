from typing import Any

import py_trees


def safe_get_blackboard(blackboard, key: str, default_value: Any = None):
    """安全获取黑板数据，支持默认值"""
    try:
        blackboard.register_key(key=key, access=py_trees.common.Access.READ)
        return blackboard.get(key)
    except KeyError:
        return default_value


def safe_set_blackboard(blackboard, key: str, value) -> bool:
    blackboard.register_key(key=key, access=py_trees.common.Access.WRITE)
    return blackboard.set(key, value)
