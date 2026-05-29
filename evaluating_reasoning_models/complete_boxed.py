def has_complete_boxed_answer(text):
    start = text.rfind(r"\boxed")
    if start == -1:
        return False

    brace_start = text.find("{", start)
    if brace_start == -1:
        return False

    depth = 0
    for ch in text[brace_start:]:
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return True

    return False