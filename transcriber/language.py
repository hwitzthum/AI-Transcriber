"""Language code normalization shared between the UI and tests.

Whisper returns full names ("french"); Deepgram returns ISO-639-1 codes
("fr") and the special sentinel "multi" for Nova-3 multilingual. This
module produces a canonical 2-letter code so the UI can compare against
the user's selection without each call site reinventing the mapping.
"""

# ISO-639-1 two-letter codes. The fast path used to accept *any* two-alpha
# string, which let nonsense values like "zz" through and produced spurious
# language-mismatch warnings. Restricting to the actual ISO-639-1 set blocks
# that without losing real codes — the UI already shows a fallback when this
# returns None, so anything not on this list surfaces as "raw value" rather
# than disappearing.
_ISO_639_1_CODES = frozenset({
    "aa", "ab", "ae", "af", "ak", "am", "an", "ar", "as", "av", "ay", "az",
    "ba", "be", "bg", "bh", "bi", "bm", "bn", "bo", "br", "bs",
    "ca", "ce", "ch", "co", "cr", "cs", "cu", "cv", "cy",
    "da", "de", "dv", "dz",
    "ee", "el", "en", "eo", "es", "et", "eu",
    "fa", "ff", "fi", "fj", "fo", "fr", "fy",
    "ga", "gd", "gl", "gn", "gu", "gv",
    "ha", "he", "hi", "ho", "hr", "ht", "hu", "hy", "hz",
    "ia", "id", "ie", "ig", "ii", "ik", "io", "is", "it", "iu",
    "ja", "jv",
    "ka", "kg", "ki", "kj", "kk", "kl", "km", "kn", "ko", "kr", "ks", "ku",
    "kv", "kw", "ky",
    "la", "lb", "lg", "li", "ln", "lo", "lt", "lu", "lv",
    "mg", "mh", "mi", "mk", "ml", "mn", "mr", "ms", "mt", "my",
    "na", "nb", "nd", "ne", "ng", "nl", "nn", "no", "nr", "nv", "ny",
    "oc", "oj", "om", "or", "os",
    "pa", "pi", "pl", "ps", "pt",
    "qu",
    "rm", "rn", "ro", "ru", "rw",
    "sa", "sc", "sd", "se", "sg", "si", "sk", "sl", "sm", "sn", "so", "sq",
    "sr", "ss", "st", "su", "sv", "sw",
    "ta", "te", "tg", "th", "ti", "tk", "tl", "tn", "to", "tr", "ts", "tt",
    "tw", "ty",
    "ug", "uk", "ur", "uz",
    "ve", "vi", "vo",
    "wa", "wo",
    "xh",
    "yi", "yo",
    "za", "zh", "zu",
})


# Whisper "verbose_json" reports detected language as the full English name in
# lowercase. Cover the languages this app advertises plus a handful of common
# native-name aliases users have in their notes.
_LANGUAGE_NAME_TO_ISO = {
    "english": "en",
    "german": "de",
    "deutsch": "de",
    "french": "fr",
    "français": "fr",
    "francais": "fr",
    "spanish": "es",
    "español": "es",
    "espanol": "es",
    "italian": "it",
    "italiano": "it",
    "portuguese": "pt",
    "português": "pt",
    "dutch": "nl",
    "polish": "pl",
    "russian": "ru",
    "japanese": "ja",
    "chinese": "zh",
    "korean": "ko",
    "arabic": "ar",
    "turkish": "tr",
    "swedish": "sv",
    "norwegian": "no",
    "danish": "da",
    "finnish": "fi",
    "czech": "cs",
    "greek": "el",
    "hindi": "hi",
}


# Sentinel returned by Deepgram Nova-3 multilingual to indicate "multiple
# languages were detected and handled". Treat it as not-comparable: the UI
# shows the value as-is and skips the mismatch check.
MULTI_SENTINEL = "multi"


def normalize_language_to_iso(value: str | None) -> str | None:
    """Return a 2-letter ISO-639-1 code for a Whisper/Deepgram language string.

    Accepts ISO codes ("fr", "fr-FR") and full names ("French", "Deutsch").
    Returns ``None`` for unknown values so callers can fall back to displaying
    the raw string. Returns ``"multi"`` unchanged so callers can detect the
    Deepgram multilingual sentinel without it being mistaken for a real code.
    """
    if not value:
        return None
    v = value.strip().lower()
    if not v:
        return None
    if v == MULTI_SENTINEL:
        return MULTI_SENTINEL
    # Strip locale suffix ("fr-FR" → "fr") and accept only known ISO-639-1.
    head = v.split("-", 1)[0]
    if len(head) == 2 and head.isalpha() and head in _ISO_639_1_CODES:
        return head
    return _LANGUAGE_NAME_TO_ISO.get(v)