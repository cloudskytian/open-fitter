import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from typing import Optional


def get_source_label(transition_label: str, config_data: Optional[dict]) -> Optional[str]:
    if config_data is None:
        return None
    transition_sets = config_data.get('blend_shape_transition_sets', [])
    if not transition_sets:
        return None
    for transition_set in transition_sets:
        if transition_set.get('label', '') == transition_label:
            return transition_set.get('source_label', '')
    return None
