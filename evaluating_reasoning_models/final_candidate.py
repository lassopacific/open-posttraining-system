from evaluating_reasoning_models.get_last_boxed_answer import get_last_boxed
import re
RE_NUMBER = re.compile(r"-?(?:\d+/\d+|\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)")


def extract_final_candidate(text, fallback="number_then_full"):
    
    ## ok, so this is the default value--if nothing matched
    result = ""

    if text:
        ## last boxed, expression --- if present
        boxed = get_last_boxed(text.strip())
        if boxed:
            result = boxed.strip().strip("#,$ ") ## removing spaces, hashes and dollar signs

        elif fallback in ("number_then_full", "number_only"): ## if there was nothing in boxed, ---then fallback
            m = RE_NUMBER.findall(text)
            if m:
                ## use last number
                result = m[-1]
            elif fallback == "number_then_full":
                ## return full text, if no number found
                return text

    return result 
        