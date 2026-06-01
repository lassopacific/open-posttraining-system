def get_last_boxed(text):
    boxed_start_idx = text.rfind(r"\boxed")

    if boxed_start_idx == -1:
        return None

    current_idx = boxed_start_idx + len(r"\boxed")

    # skip whitespace
    while current_idx < len(text) and text[current_idx].isspace():
        current_idx += 1

    # must start with {
    if current_idx >= len(text) or text[current_idx] != "{":
        return None

    current_idx += 1
    brace_depth = 1
    content_start = current_idx

    while current_idx < len(text) and brace_depth > 0:
        char = text[current_idx]

        if char == "{":
            brace_depth += 1
        elif char == "}":
            brace_depth -= 1

        current_idx += 1

    # unmatched braces
    if brace_depth != 0:
        return None

    return text[content_start:current_idx - 1]
