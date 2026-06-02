import re

# Common LaTeX cleanup replacements
LATEX_FIXES = [
    (r"\\left\s*", ""),          # remove \left
    (r"\\right\s*", ""),         # remove \right
    (r"\\,|\\!|\\;|\\:", ""),    # remove latex spacing commands
    (r"\\cdot", "*"),            # convert multiplication symbol
    (r"\u00B7|\u00D7", "*"),     # unicode multiplication -> *
    (r"\\\^\\circ", ""),         # remove degree notation
    (r"\\dfrac", r"\\frac"),     # normalize dfrac -> frac
    (r"\\tfrac", r"\\frac"),     # normalize tfrac -> frac
    (r"Â°", ""),                 # remove corrupted degree symbol
]

# Remove chat special tokens like <|assistant|>
RE_SPECIAL = re.compile(r"<\|[^>]+?\|>")


def normalize_text(text):

    # Handle empty input
    if not text:
        return ""

    # Remove special tokens and surrounding whitespace
    text = RE_SPECIAL.sub("", text).strip()

    # Unicode superscript mapping
    SUPERSCRIPT_MAP = {
    "⁰": "0",
    "¹": "1",
    "²": "2",
    "³": "3",
    "⁴": "4",
    "⁵": "5",
    "⁶": "6",
    "⁷": "7",
    "⁸": "8",
    "⁹": "9",
    "⁺": "+",
    "⁻": "-",
    "⁽": "(",
    "⁾": ")",
}

    # Remove leading labels like "A: answer"
    match = re.match(r"^[A-Za-z]\s*[.:]\s*(.+)$", text)
    if match:
        text = match.group(1)

    # Remove LaTeX degree notation
    text = re.sub(r"\^\s*\{\s*\\circ\s*\}", "", text)
    text = re.sub(r"\^\s*\\circ", "", text)
    text = text.replace("Â°", "")

    # Unwrap \text{...} if entire string is wrapped
    match = re.match(r"^\\text\{(?P<x>.+?)\}$", text)
    if match:
        text = match.group("x")

    # Remove latex bracket wrappers \( \) \[ \]
    text = re.sub(r"\\\(|\\\)|\\\[|\\\]", "", text)

    # Apply generic latex cleanup rules
    for pat, rep in LATEX_FIXES:
        text = re.sub(pat, rep, text)

    # Convert unicode superscripts into normal exponent strings
    def convert_superscripts(s, base=None):

        converted = "".join(
            SUPERSCRIPT_MAP[ch] if ch in SUPERSCRIPT_MAP else ch
            for ch in s
        )

        if base is None:
            return converted

        return f"{base}**{converted}"

    # Convert cases like x² -> x**2
    text = re.sub(
        r"([0-9A-Za-z\)\]\}])([â°Â¹Â²Â³â´âµâ¶â·â¸â¹âºâ»]+)",
        lambda m: convert_superscripts(
            m.group(2),
            base=m.group(1),
        ),
        text,
    )

    # Convert standalone superscripts
    text = convert_superscripts(text)

    # Remove percentage and dollar symbols
    text = text.replace("\\%", "%").replace("$", "").replace("%", "")

    # Convert \sqrt{a} -> sqrt(a)
    text = re.sub(
        r"\\sqrt\s*\{([^}]*)\}",
        lambda match: f"sqrt({match.group(1)})",
        text,
    )

    # Convert \sqrt x -> sqrt(x)
    text = re.sub(
        r"\\sqrt\s+([^\\\s{}]+)",
        lambda match: f"sqrt({match.group(1)})",
        text,
    )

    # Convert \frac{a}{b} -> (a)/(b)
    text = re.sub(
        r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}",
        lambda match: f"({match.group(1)})/({match.group(2)})",
        text,
    )

    # Convert \frac a b -> (a)/(b)
    text = re.sub(
        r"\\frac\s+([^\s{}]+)\s+([^\s{}]+)",
        lambda match: f"({match.group(1)})/({match.group(2)})",
        text,
    )

    # Convert ^ -> python exponent **
    text = text.replace("^", "**")

    # Convert mixed numbers: 2 1/2 -> 2+1/2
    text = re.sub(
        r"(?<=\d)\s+(\d+/\d+)",
        lambda match: "+" + match.group(1),
        text,
    )

    # Remove commas from large numbers: 1,234 -> 1234
    text = re.sub(
        r"(?<=\d),(?=\d\d\d(\D|$))",
        "",
        text,
    )

    # Final cleanup
    return text.replace("{", "").replace("}", "").strip().lower()