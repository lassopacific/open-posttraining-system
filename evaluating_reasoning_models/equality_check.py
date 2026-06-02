from sympy import simplify
from evaluating_reasoning_models.sympy_parser import sympy_parser

def equality_check(expr_gtruth, expr_pred):
    ## checking if both of the expressions are, exactly the same string
    if expr_gtruth ==  expr_pred:
        return True

    ## parsing both expressions into sympy parser--- returns None if parsing fails
    gtruth, pred = sympy_parser(expr_gtruth), sympy_parser(expr_pred)


    ## if both expression were parsed successfully, now try---> symbolic  comparison
    if gtruth is not None and pred is not None:
        try:
        ## here if the difference is 0, they are equivalent
            return simplify(gtruth - pred) == 0

        except (SympifyError, TypeError):
            pass
    
    return  False


