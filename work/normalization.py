"""
Standard-Aware Normalization Module
- Country: ISO 3166-1 alpha-2
- Language: BCP 47 (alpha-2 with alpha-3 fallback)
- Genre / contentRating helpers stubbed for MovieLens extension

This module is post-hoc: it accepts free-text LLM output and maps it
to a canonical standard code, returning (canonical, was_compliant).
`was_compliant` is True iff the raw value was already in canonical form.
"""

import re
import pycountry

INVALID_TOKENS = {"", "country", "language", "n/a", "na", "none", "null", "unknown"}

COUNTRY_ALIASES = {
    "USA": "US", "U.S.A": "US", "U.S.": "US", "US": "US",
    "United States": "US", "United States of America": "US", "America": "US",
    "UK": "GB", "U.K.": "GB", "United Kingdom": "GB", "England": "GB",
    "Britain": "GB", "Great Britain": "GB", "Scotland": "GB", "Wales": "GB",
    "Russia": "RU", "Soviet Union": "RU", "USSR": "RU",
    "South Korea": "KR", "Korea": "KR", "Korea, South": "KR",
    "North Korea": "KP", "Korea, North": "KP",
    "Czech Republic": "CZ", "Czechia": "CZ",
    "Iran": "IR", "Persia": "IR",
    "Vietnam": "VN", "Viet Nam": "VN",
    "Taiwan": "TW", "Republic of China": "TW",
    "Hong Kong": "HK",
    "West Germany": "DE", "East Germany": "DE", "Germany": "DE",
    "Yugoslavia": "RS",
    "Burma": "MM", "Myanmar": "MM",
}

LANG_ALIASES = {
    "English": "en", "American English": "en", "British English": "en",
    "Spanish": "es", "Castilian": "es",
    "French": "fr", "German": "de", "Italian": "it",
    "Japanese": "ja", "Korean": "ko",
    "Chinese": "zh", "Mandarin": "zh", "Cantonese": "zh-yue",
    "Russian": "ru", "Portuguese": "pt", "Brazilian Portuguese": "pt-BR",
    "Hindi": "hi", "Arabic": "ar", "Hebrew": "he", "Yiddish": "yi",
    "Dutch": "nl", "Swedish": "sv", "Norwegian": "no", "Danish": "da",
    "Finnish": "fi", "Polish": "pl", "Czech": "cs", "Greek": "el",
    "Turkish": "tr", "Thai": "th", "Vietnamese": "vi", "Indonesian": "id",
    "Latin": "la", "Sign Language": "sgn", "American Sign Language": "ase",
    "Silent": "zxx",
}

ISO_3166_ALPHA2 = {c.alpha_2 for c in pycountry.countries}
ISO_639_ALPHA2 = {l.alpha_2 for l in pycountry.languages if hasattr(l, "alpha_2")}
ISO_639_ALPHA3 = {l.alpha_3 for l in pycountry.languages if hasattr(l, "alpha_3")}


def _is_invalid(s):
    return (not isinstance(s, str)) or s.strip().lower() in INVALID_TOKENS


def normalize_country(raw):
    """Map free-text country to ISO 3166-1 alpha-2.
    Returns (canonical_code or None, was_already_compliant: bool).
    """
    if _is_invalid(raw):
        return None, False
    s = raw.strip()
    # already alpha-2 ISO?
    if len(s) == 2 and s.upper() in ISO_3166_ALPHA2:
        return s.upper(), True
    # alias?
    if s in COUNTRY_ALIASES:
        return COUNTRY_ALIASES[s], False
    # pycountry direct lookup
    try:
        c = pycountry.countries.lookup(s)
        return c.alpha_2, False
    except LookupError:
        pass
    # split on common multi-country separators and try first
    for sep in [",", "/", "|", ";", " and ", "&"]:
        if sep in s:
            head = s.split(sep)[0].strip()
            try:
                c = pycountry.countries.lookup(head)
                return c.alpha_2, False
            except LookupError:
                if head in COUNTRY_ALIASES:
                    return COUNTRY_ALIASES[head], False
    return None, False


def normalize_language(raw):
    """Map free-text language to BCP 47 short tag.
    Returns (canonical_tag or None, was_already_compliant: bool).
    """
    if _is_invalid(raw):
        return None, False
    s = raw.strip()
    # reject obvious column-shift errors where a country leaked into language field
    if s in COUNTRY_ALIASES or (len(s) == 2 and s.upper() in ISO_3166_ALPHA2):
        return None, False
    # 2 char alpha-2 already?
    if len(s) == 2 and s.lower() in ISO_639_ALPHA2:
        return s.lower(), True
    # alias?
    if s in LANG_ALIASES:
        return LANG_ALIASES[s], False
    # pycountry direct lookup
    try:
        l = pycountry.languages.lookup(s)
        return getattr(l, "alpha_2", None) or getattr(l, "alpha_3", None), False
    except LookupError:
        pass
    # try first segment of multi-language fields
    for sep in [",", "/", "|", ";", " and ", "&"]:
        if sep in s:
            head = s.split(sep)[0].strip()
            if head in LANG_ALIASES:
                return LANG_ALIASES[head], False
            try:
                l = pycountry.languages.lookup(head)
                return getattr(l, "alpha_2", None) or getattr(l, "alpha_3", None), False
            except LookupError:
                continue
    return None, False


def normalize_year(raw):
    """4-digit year per ISO 8601 datePublished."""
    if raw is None:
        return None, False
    try:
        y = int(float(raw))
        if 1880 <= y <= 2030:
            return str(y), True
    except (ValueError, TypeError):
        pass
    m = re.search(r"\b(18[8-9]\d|19\d\d|20[0-2]\d)\b", str(raw))
    if m:
        return m.group(1), False
    return None, False


def normalize_director(raw):
    """No global standard for personal names; just whitespace cleanup.
    Compliance is defined as 'non-empty, non-placeholder'."""
    if _is_invalid(raw):
        return None, False
    s = re.sub(r"\s+", " ", raw.strip())
    return s, True


# unified API
NORMALIZERS = {
    "country": normalize_country,
    "language": normalize_language,
    "year": normalize_year,
    "director": normalize_director,
}


def normalize_record(record):
    """Apply all normalizers to a {field: raw_value} record.
    Returns (normalized_dict, compliance_dict)."""
    out, comp = {}, {}
    for field, raw in record.items():
        if field in NORMALIZERS:
            canonical, was_compliant = NORMALIZERS[field](raw)
            out[field] = canonical
            comp[field] = was_compliant
        else:
            out[field] = raw
            comp[field] = None
    return out, comp


if __name__ == "__main__":
    samples = [
        {"director": "Jim Jarmusch", "country": "USA", "language": "English", "year": 1986},
        {"director": "Mike Nichols", "country": "United States", "language": "English", "year": 1970.0},
        {"director": "Akira Kurosawa", "country": "Japan", "language": "Japanese", "year": 1954},
        {"director": "?", "country": "country", "language": "USA", "year": "n/a"},
        {"director": "Bong Joon-ho", "country": "South Korea", "language": "Korean", "year": 2019},
    ]
    for s in samples:
        norm, comp = normalize_record(s)
        print("RAW :", s)
        print("NORM:", norm)
        print("COMP:", comp)
        print()
