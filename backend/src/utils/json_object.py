import json
from typing import Any

class JSONObject:
    def __init__(self, data: Any):
        if isinstance(data, dict):
            for key, value in data.items():
                setattr(self, key, JSONObject(value))
        elif isinstance(data, list):
            self._list = [JSONObject(item) for item in data]
        else:
            self._value = data

    def __getitem__(self, key):
        if hasattr(self, "_list"):
            return self._list[key]
        return getattr(self, key)

    def __iter__(self):
        if hasattr(self, "_list"):
            return iter(self._list)
        return iter(self.__dict__)

    def __len__(self):
        if hasattr(self, "_list"):
            return len(self._list)
        return len(self.__dict__)

    def __repr__(self):
        if hasattr(self, "_list"):
            return repr(self._list)
        if hasattr(self, "_value"):
            return repr(self._value)
        return repr(self.__dict__)

    def to_dict(self):
        if hasattr(self, "_list"):
            return [item.to_dict() for item in self._list]
        if hasattr(self, "_value"):
            return self._value
        return {key: value.to_dict() for key, value in self.__dict__.items()}
    
    def get(self, key, default=None):
        return getattr(self, key, default)

    def __contains__(self, key):
        return key in self.__dict__


def load_json_as_object(filepath: str) -> JSONObject:
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return JSONObject(data)

from src.utils.json_object import JSONObject

def write_object_as_json(filepath: str, data: Any, indent: int = 2, ensure_ascii: bool = False):
    """Write a Python object (dict, JSONObject, list, etc.) to a JSON file."""
    with open(filepath, "w", encoding="utf-8") as f:
        if isinstance(data, JSONObject):
            json.dump(data.to_dict(), f, indent=indent, ensure_ascii=ensure_ascii)
        else:
            # 👇 this ensures we recursively turn JSONObject inside dict/list into dicts
            def convert(obj):
                if isinstance(obj, JSONObject):
                    return obj.to_dict()
                if isinstance(obj, dict):
                    return {k: convert(v) for k, v in obj.items()}
                if isinstance(obj, list):
                    return [convert(v) for v in obj]
                return obj

            json.dump(convert(data), f, indent=indent, ensure_ascii=ensure_ascii)
