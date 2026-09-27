

"""
STEP 2 — BUILD THE EXPERIMENT for ONE story (any Python with numpy + soundfile;
no Kokoro). Makes experiment/index.html: participants type their name, choose
a version, listen sentence by sentence in any order, and at the end can
download a CSV, review/edit their answers, or restart with another version.
Reads the pages saved by run_generation.py in outputs/.
"""
from experiment import build_experiment

TEST_ID    = "story1_full"
SUBMIT_URL = "https://script.google.com/macros/s/AKfycbwqf8azLBPy0_GLpCBhqXidUONqDuJVpvL4dNhMBpRAIpxhZgT5zfCzi93NnwkCw-jt/exec"      # your Google Apps Script /exec URL
LANG       = "en"      # interface language: "it" or "en"
TEXT_AFTER_LISTEN = True   # a sentence's text appears only after it was heard once
EMBED_AUDIO = True     # True: one self-contained file. False: page + audio/ folder

story_number = 1

# versions to offer: id stored with the answers -> outputs/<name>.html
VERSIONS = {
    "native":       f"output_online/story{story_number}__clean.html",
    "consistent":   f"output_online/story{story_number}__consistent_dur_forced_native.html",
    "inconsistent": f"output_online/story{story_number}__less_consistent_dur_forced_native.html",
}

# what participants see for each version
VERSION_LABELS = {
    "native":       "native",
    "consistent":   "errors_type1",
    "inconsistent": "errors_type2",
}

build_experiment(
    VERSIONS, out_dir="experiment", out_html=f"story_{story_number}.html",
    test_id=TEST_ID, version_labels=VERSION_LABELS, submit_url=SUBMIT_URL,
    text_after_listen=TEXT_AFTER_LISTEN, lang=LANG, embed_audio=EMBED_AUDIO,
)