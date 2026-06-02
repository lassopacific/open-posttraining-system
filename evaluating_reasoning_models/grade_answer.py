from evaluating_reasoning_models.split_into_parts import split_into_parts
from evaluating_reasoning_models.normalize_text import normalize_text
from evaluating_reasoning_models.equality_check import equality_check

def grade_answer(pd_text, gdt_text):
    result = False   ## this will be the default outcome,,--if the checks fail here

    ## only continuing if both of the inputs  are non-empty strings
    if pd_text is not None and gdt_text is not None:
        gdt_parts = split_into_parts(normalize_text(gdt_text)) ##  breaking gdt_text ---> into comparable parts

        pd_parts  = split_into_parts(normalize_text(pd_text)) ## braking  prediction text ----> into comparable parts


        ## ensuring both sides --have same number of valid parts
        if (gdt_parts and pd_parts and len(gdt_parts) == len(pd_parts)):
            result = all(equality_check(gt, pred) for gt, pred in zip(gdt_parts, pd_parts))


    return result  ## true only when all the checks are passed..