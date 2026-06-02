import sympy.parsing.sympy_parser as spp
from sympy.core.sympify import SympifyError
from tokenize import TokenError
from sympy.polys.polyerrors import PolynomialError

def sympy_parser(expr):

    # 1. Guard rails against memory exhaustion and null inputs--badly trained models generate some --garbage
    if not expr or len(expr) > 2000:
        return None

    try:
        # 2. Parse and return the evaluated expression
        return spp.parse_expr(
            expr,
            transformations=(
                *spp.standard_transformations,  ##stardard  transformation for handling parenthesis
                spp.implicit_multiplication_application, ## aloweing omitted  multiplication  symbols
            ),
            evaluate=True
        )
    except (
        SympifyError, 
        SyntaxError, 
        TypeError, 
        AttributeError, 
        IndexError, 
        TokenError, 
        ValueError, 
        PolynomialError,
        ZeroDivisionError  # Added to catch literal divisions like '1/0' during evaluation
    ):
        return None