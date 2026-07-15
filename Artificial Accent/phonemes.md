

# input of the model

```
text → misaki (KPipeline, lang_code='i')
           └─ espeak-ng  →  raw IPA
           └─ misaki post-processing (normalize → map to Kokoro symbol set)
       → phoneme string → token IDs → model
```

- **misaki** is Kokoro's grapheme-to-phoneme (G2P) engine. For Italian it wraps `espeak-ng` and maps the output into misaki's own symbol set. Author's note in misaki: the symbols are *"intended as input tokens for neural networks,"* not strict IPA, so a few glyphs are repurposed.

## vocab dictionary
Each phoneme character maps to one integer via the vocab table below.

| ID | Symbol | Description |
|----|--------|-------------|
| 1 | `;` | semicolon |
| 2 | `:` | colon |
| 3 | `,` | comma |
| 4 | `.` | period |
| 5 | `!` | exclamation mark |
| 6 | `?` | question mark |
| 9 | `—` | em dash |
| 10 | `…` | ellipsis |
| 11 | `"` | straight quote |
| 12 | `(` | left paren |
| 13 | `)` | right paren |
| 14 | `“` | left curly quote |
| 15 | `”` | right curly quote |
| 16 | ` ` | space |
| 17 | `̃` | combining tilde (nasalization) |
| 18 | `ʣ` | voiced alveolar affricate /d͡z/ |
| 19 | `ʥ` | voiced alveolo-palatal affricate /d͡ʑ/ |
| 20 | `ʦ` | voiceless alveolar affricate /t͡s/ |
| 21 | `ʨ` | voiceless alveolo-palatal affricate /t͡ɕ/ |
| 22 | `ᵝ` | misaki-internal reduced marker (not IPA) |
| 23 | `ꭧ` | retroflex affricate (U+AB67) |
| 24 | `A` | misaki: FACE vowel → /eɪ/ (not IPA) |
| 25 | `I` | misaki: PRICE vowel → /aɪ/ (not IPA) |
| 31 | `O` | misaki: GOAT vowel US → /oʊ/ (not IPA) |
| 33 | `Q` | misaki: GOAT vowel GB → /əʊ/ (not IPA) |
| 35 | `S` | misaki-internal token (not IPA /s/) |
| 36 | `T` | misaki-internal token (not IPA /t/) |
| 39 | `W` | misaki: MOUTH vowel → /aʊ/ (not IPA) |
| 41 | `Y` | misaki: CHOICE vowel → /ɔɪ/ (not IPA) |
| 42 | `ᵊ` | reduced/muted schwa |
| 43 | `a` | open front unrounded vowel |
| 44 | `b` | voiced bilabial plosive |
| 45 | `c` | voiceless palatal plosive |
| 46 | `d` | voiced alveolar plosive |
| 47 | `e` | close-mid front unrounded vowel |
| 48 | `f` | voiceless labiodental fricative |
| 50 | `h` | voiceless glottal fricative |
| 51 | `i` | close front unrounded vowel |
| 52 | `j` | palatal approximant (the "y" glide) |
| 53 | `k` | voiceless velar plosive |
| 54 | `l` | alveolar lateral approximant |
| 55 | `m` | bilabial nasal |
| 56 | `n` | alveolar nasal |
| 57 | `o` | close-mid back rounded vowel |
| 58 | `p` | voiceless bilabial plosive |
| 59 | `q` | voiceless uvular plosive |
| 60 | `r` | alveolar trill |
| 61 | `s` | voiceless alveolar fricative |
| 62 | `t` | voiceless alveolar plosive |
| 63 | `u` | close back rounded vowel |
| 64 | `v` | voiced labiodental fricative |
| 65 | `w` | labial-velar approximant |
| 66 | `x` | voiceless velar fricative |
| 67 | `y` | close front rounded vowel |
| 68 | `z` | voiced alveolar fricative |
| 69 | `ɑ` | open back unrounded vowel |
| 70 | `ɐ` | near-open central vowel |
| 71 | `ɒ` | open back rounded vowel |
| 72 | `æ` | near-open front unrounded vowel |
| 75 | `β` | voiced bilabial fricative |
| 76 | `ɔ` | open-mid back rounded vowel |
| 77 | `ɕ` | voiceless alveolo-palatal fricative |
| 78 | `ç` | voiceless palatal fricative |
| 80 | `ɖ` | voiced retroflex plosive |
| 81 | `ð` | voiced dental fricative |
| 82 | `ʤ` | voiced postalveolar affricate /d͡ʒ/ |
| 83 | `ə` | schwa |
| 85 | `ɚ` | r-colored schwa |
| 86 | `ɛ` | open-mid front unrounded vowel |
| 87 | `ɜ` | open-mid central unrounded vowel |
| 90 | `ɟ` | voiced palatal plosive |
| 92 | `ɡ` | voiced velar plosive (U+0261) |
| 99 | `ɥ` | labial-palatal approximant |
| 101 | `ɨ` | close central unrounded vowel |
| 102 | `ɪ` | near-close near-front unrounded vowel |
| 103 | `ʝ` | voiced palatal fricative |
| 110 | `ɯ` | close back unrounded vowel |
| 111 | `ɰ` | velar approximant |
| 112 | `ŋ` | velar nasal |
| 113 | `ɳ` | retroflex nasal |
| 114 | `ɲ` | palatal nasal |
| 115 | `ɴ` | uvular nasal |
| 116 | `ø` | close-mid front rounded vowel |
| 118 | `ɸ` | voiceless bilabial fricative |
| 119 | `θ` | voiceless dental fricative |
| 120 | `œ` | open-mid front rounded vowel |
| 123 | `ɹ` | alveolar approximant |
| 125 | `ɾ` | alveolar tap |
| 126 | `ɻ` | retroflex approximant |
| 128 | `ʁ` | voiced uvular fricative |
| 129 | `ɽ` | retroflex tap |
| 130 | `ʂ` | voiceless retroflex fricative |
| 131 | `ʃ` | voiceless postalveolar fricative |
| 132 | `ʈ` | voiceless retroflex plosive |
| 133 | `ʧ` | voiceless postalveolar affricate /t͡ʃ/ |
| 135 | `ʊ` | near-close near-back rounded vowel |
| 136 | `ʋ` | labiodental approximant |
| 138 | `ʌ` | open-mid back unrounded vowel |
| 139 | `ɣ` | voiced velar fricative |
| 140 | `ɤ` | close-mid back unrounded vowel |
| 142 | `χ` | voiceless uvular fricative |
| 143 | `ʎ` | palatal lateral approximant |
| 147 | `ʒ` | voiced postalveolar fricative |
| 148 | `ʔ` | glottal stop |
| 156 | `ˈ` | primary stress |
| 157 | `ˌ` | secondary stress |
| 158 | `ː` | length mark (gemination) |
| 162 | `ʰ` | aspiration |
| 164 | `ʲ` | palatalization |
| 169 | `↓` | downstep / pitch fall |
| 171 | `→` | level tone |
| 172 | `↗` | rising intonation |
| 173 | `↘` | falling intonation |
| 177 | `ᵻ` | vowel between /ə/ and /ɪ/ |

That's all 114 entries from your dict. the gaps you're seeing are genuine unused slots in the model(7–8, 26–30,...) 

- **For stimulus manipulation you can bypass G2P and feed a phoneme string directly**, as long as every character exists in the vocab. This is the intended lever for controlled edits (lengthening, stress shifts, segment substitution, etc.).

Some vocab entries are **not IPA** and must not be treated as their look-alike sound. This matters most when hand-editing token sequences:

| Token | ID | Meaning | ⚠️ Not to be confused with |
|-------|----|---------|-----|
| `ᵊ` | 42 | reduced/muted schwa | weaker than `ə` (83) |
| `ᵻ` | 177 | vowel between `ə` and `ɪ` (English -es endings) | — |
| `ᵝ` | 22 | misaki-internal reduced marker | not the fricative `β` (75) |
| `ꭧ` | 23 | retroflex affricate (U+AB67) | — |

Also note `j` = the "y" glide (`/j/`), and `ɡ` is U+0261, not ASCII "g".


## Available (non-Italian phonemes you *can* insert)

**Fricatives:** `ɸ` β `β` (bilabial), `θ` `ð` (dental), `ç` `ʝ` (palatal), `x` `ɣ` (velar), `χ` `ʁ` (uvular), `ʂ` (retroflex, voiceless), `ɕ` (alveolo-palatal, voiceless), `ʒ` (postalveolar), `h`.

**Plosives / affricates:** `c` `ɟ` (palatal), `q` (uvular), `ʔ` (glottal stop); affricate ligatures `ʨ` `ʥ` (alveolo-palatal), plus `ꭧ` (retroflex).

**Retroflex set:** `ʈ` `ɖ` `ɳ` `ʂ` `ɽ` `ɻ` (+ affricate `ꭧ`).

**Nasals:** `ɴ` (uvular), plus the Italian `ɳ ɲ ŋ`.

**Approximants / rhotics:** `ɹ` `ɻ` (r-type), `ʋ` (labiodental), `ɥ` (labial-palatal), `ɰ` (velar), `ɰ`.

**Vowels:** open `ɑ` `ɐ` `ɒ` `æ`; front rounded `ø` `œ`; close `ɨ` `ɯ`; near-close `ɪ` `ʊ`; mid/central `ə` `ɚ` `ɜ` `ʌ` `ɤ`.

**Diacritics / suprasegmentals:** nasalization (combining) `̃` (17), aspiration `ʰ` (162), palatalization `ʲ` (164), and intonation/tone arrows `↓ → ↗ ↘` (169–173).

### NOT available (phonemes absent from the vocab)

You cannot represent these; you must approximate or substitute.

- **Voiced retroflex fricative `ʐ`** — absent, even though voiceless `ʂ` is present (Mandarin/Polish/Russian "r/rz").
- **Voiced alveolo-palatal fricative `ʑ`** — absent, even though voiceless `ɕ` is present.
- **Pharyngeals `ħ ʕ`** and **voiced glottal `ɦ`** (Arabic; Hindi/Czech `ɦ`).
- **Lateral fricatives `ɬ ɮ`** (Welsh, Zulu, Nahuatl).
- **Nasals/laterals `ɱ` (labiodental), `ɭ` (retroflex lateral), `ʟ` (velar lateral).**
- **Trills `ʙ ʀ`** and **labiodental flap `ⱱ`** (only taps `ɾ ɽ` and trill letter `r` exist).
- **Plosives `ɢ` (voiced uvular), `ʡ` (epiglottal).**
- **Rounded central/front vowels `ʉ ɵ ɘ ɞ ʏ ɶ`**, and **`ɝ`** (r-colored `ɜ`; only `ɚ` exists).
- **Labialization `ʷ`** — absent, though `ʲ` is present (blocks e.g. `kʷ`, Russian/Irish contrasts).
- **Other diacritics:** velarization `ˠ`, pharyngealization `ˤ`, ejective `ʼ`, breathy/voiced aspiration `ʱ` (only voiceless `ʰ` exists), syllabicity `̩`, half-long `ˑ`, tie bar (affricates use precomposed ligatures instead).
- **Whole classes:** clicks (`ʘ ǀ ǃ ǂ ǁ`), implosives (`ɓ ɗ ʄ ɠ ʛ`), and `ʍ`.