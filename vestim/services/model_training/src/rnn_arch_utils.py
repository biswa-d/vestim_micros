import re
from typing import List, Union


def parse_layer_sizes(val: Union[str, List[int], None], fallback_hidden: int = None, fallback_layers: int = None) -> List[int]:
    """
    Parse a layer sizes specification into a list of ints.
    Accepts:
    - Comma-separated string like "64,32,16"
    - List/tuple of ints [64,32,16]
    - None + fallbacks (hidden, layers) -> [hidden]*layers
    Returns list of positive ints.
    """
    if val is None:
        if fallback_hidden is None or fallback_layers is None:
            raise ValueError("parse_layer_sizes requires either 'val' or both fallbacks (hidden, layers)")
        return [int(fallback_hidden)] * int(fallback_layers)

    if isinstance(val, (list, tuple)):
        sizes = [int(x) for x in val]
    elif isinstance(val, str):
        # Remove brackets/spaces
        s = val.strip().strip('[]()')
        if not s:
            raise ValueError("Empty layer sizes string")
        parts = re.split(r"\s*,\s*", s)
        sizes = [int(p) for p in parts]
    else:
        raise TypeError(f"Unsupported layer sizes type: {type(val)}")

    if any(x <= 0 for x in sizes):
        raise ValueError(f"All layer sizes must be positive, got {sizes}")
    return sizes


def all_equal(sizes: List[int]) -> bool:
    return all(x == sizes[0] for x in sizes)
