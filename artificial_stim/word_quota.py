"""Compatibility: the code now lives in accent_tools.py. Old notebooks that import word_quota keep working."""
from accent_tools import *                                   # noqa: F401,F403
from accent_tools import (manipulate_with_quota, select_quota_words, adjust_phonemes, normalize_ipa,  # noqa: F401
                          spell_words, _word_spans, NORMALIZE_DEFAULT)
