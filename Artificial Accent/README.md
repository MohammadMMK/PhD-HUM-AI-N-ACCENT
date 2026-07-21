

# input of the model

```
text → misaki (KPipeline, lang_code='i')
           └─ espeak-ng  →  raw IPA
           └─ misaki post-processing (normalize → map to Kokoro symbol set)
       → phoneme string → token IDs → model
```

- **misaki** is Kokoro's grapheme-to-phoneme (G2P) engine. For Italian it wraps `espeak-ng` and maps the output into misaki's own symbol set. Author's note in misaki: the symbols are *"intended as input tokens for neural networks,"* not strict IPA, so a few glyphs are repurposed.

- Each phoneme character maps to one integer via the vocab dictionary ([see here](vocab_dict_kokoro.md)). Also check available and not available foreign phonemes.

- **For stimulus manipulation you can bypass G2P and feed a phoneme string directly**, as long as every character exists in the vocab. This is the intended lever for controlled edits (lengthening, stress shifts, segment substitution, etc.).



