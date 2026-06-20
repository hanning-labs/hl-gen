"""Deterministic code-switching ratio measurement using lingua-language-detector."""

from __future__ import annotations

import logging
import re
from functools import lru_cache

log = logging.getLogger(__name__)

# Splits text into tokens for lingua classification:
# - Each CJK / Kana / Hangul character is its own token (ideographic scripts have no spaces)
# - All other scripts: whitespace-delimited words
#
# The word alternative explicitly excludes CJK/Kana/Hangul so that adjacent runs like
# "tired啊" are split into ["tired", "啊"] rather than matched as one token by \w+.
_CJK_KANA_HANGUL = (
    r"一-鿿"   # CJK Unified Ideographs
    r"㐀-䶿"   # CJK Extension A
    r"぀-ヿ"   # Hiragana + Katakana
    r"가-힣"   # Hangul Syllables
)
_TOKEN_RE = re.compile(
    rf"[{_CJK_KANA_HANGUL}]"             # single ideographic token
    rf"|[^\s\W{_CJK_KANA_HANGUL}]+"      # non-CJK word token
)

# Maps lowercase language names to lingua Language enum member names.
# Where multiple spoken varieties share a writing system (e.g. Cantonese / Mandarin),
# they map to the same lingua label.
_LINGUA_NAMES: dict[str, str] = {
    "afrikaans": "AFRIKAANS",
    "arabic": "ARABIC",
    "azerbaijani": "AZERBAIJANI",
    "basque": "BASQUE",
    "belarusian": "BELARUSIAN",
    "bengali": "BENGALI",
    "bosnian": "BOSNIAN",
    "bulgarian": "BULGARIAN",
    "cantonese": "CHINESE",
    "catalan": "CATALAN",
    "chinese": "CHINESE",
    "croatian": "CROATIAN",
    "czech": "CZECH",
    "danish": "DANISH",
    "dutch": "DUTCH",
    "english": "ENGLISH",
    "estonian": "ESTONIAN",
    "finnish": "FINNISH",
    "french": "FRENCH",
    "georgian": "GEORGIAN",
    "german": "GERMAN",
    "greek": "GREEK",
    "gujarati": "GUJARATI",
    "hebrew": "HEBREW",
    "hindi": "HINDI",
    "hungarian": "HUNGARIAN",
    "icelandic": "ICELANDIC",
    "indonesian": "INDONESIAN",
    "irish": "IRISH",
    "italian": "ITALIAN",
    "japanese": "JAPANESE",
    "kazakh": "KAZAKH",
    "korean": "KOREAN",
    "latvian": "LATVIAN",
    "lithuanian": "LITHUANIAN",
    "macedonian": "MACEDONIAN",
    "malay": "MALAY",
    "mandarin": "CHINESE",
    "marathi": "MARATHI",
    "mongolian": "MONGOLIAN",
    "norwegian": "BOKMAL",
    "persian": "PERSIAN",
    "polish": "POLISH",
    "portuguese": "PORTUGUESE",
    "punjabi": "PUNJABI",
    "romanian": "ROMANIAN",
    "russian": "RUSSIAN",
    "serbian": "SERBIAN",
    "slovak": "SLOVAK",
    "slovene": "SLOVENE",
    "somali": "SOMALI",
    "spanish": "SPANISH",
    "swahili": "SWAHILI",
    "swedish": "SWEDISH",
    "tagalog": "TAGALOG",
    "tamil": "TAMIL",
    "telugu": "TELUGU",
    "thai": "THAI",
    "turkish": "TURKISH",
    "ukrainian": "UKRAINIAN",
    "urdu": "URDU",
    "vietnamese": "VIETNAMESE",
    "welsh": "WELSH",
    "yoruba": "YORUBA",
    "zulu": "ZULU",
}

# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def ratio_score(actual: float, target: float) -> float:
    """Score 0–10 via relative error against the target ratio.

    Error is normalised by min(target, 1-target) — the tightest achievable
    deviation from the target — so a 10 % miss on a 10 % target (100 % relative
    error) scores the same as a 50 % miss on a 50 % target.

    actual == 0 always returns 0; identical values always return 10.
    """
    if actual == 0.0:
        return 0.0
    if actual == target:
        return 10.0
    norm = min(target, 1.0 - target)
    if norm == 0.0:
        # target is 0 or 1: only a perfect match passes
        return 10.0 if actual == target else 0.0
    error = abs(actual - target)
    return max(0.0, min(10.0, 10.0 * (1.0 - error / norm)))


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    """Split into tokens: each CJK/Kana/Hangul char is its own; other scripts are word-split."""
    return _TOKEN_RE.findall(text)


@lru_cache(maxsize=32)
def _build_detector(l1_name: str, l2_name: str):
    from lingua import Language, LanguageDetectorBuilder  # type: ignore[import]
    l1 = getattr(Language, l1_name)
    l2 = getattr(Language, l2_name)
    return LanguageDetectorBuilder.from_languages(l1, l2).build()


def count_tokens(text: str, l1_lang: str, l2_lang: str) -> tuple[int, int]:
    """Return ``(l1_token_count, l2_token_count)`` via lingua-language-detector.

    Tokens that lingua cannot assign to either language (e.g. numbers, punctuation)
    are skipped and do not contribute to either count.
    """
    l1_name = _LINGUA_NAMES.get(l1_lang.lower())
    l2_name = _LINGUA_NAMES.get(l2_lang.lower())

    if not l1_name:
        log.warning("No lingua mapping for %r — L1 tokens will not be counted", l1_lang)
        l1_name = "ENGLISH"
    if not l2_name:
        log.warning("No lingua mapping for %r — L2 tokens will not be counted", l2_lang)
        l2_name = "ENGLISH"

    try:
        from lingua import Language  # type: ignore[import]
    except ImportError:
        log.error(
            "lingua-language-detector is required for ratio scoring. "
            "Install with: pip install 'code-switch-data-synthesis[tools]'"
        )
        return 0, 0

    l1_lingua = getattr(Language, l1_name)
    l2_lingua = getattr(Language, l2_name)
    detector = _build_detector(l1_name, l2_name)

    l1_count = 0
    l2_count = 0
    for token in _tokenize(text):
        lang = detector.detect_language_of(token)
        if lang == l1_lingua:
            l1_count += 1
        elif lang == l2_lingua:
            l2_count += 1

    return l1_count, l2_count
