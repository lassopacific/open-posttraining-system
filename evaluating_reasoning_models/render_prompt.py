def render_prompt(prompt):
    template = (
        "You are a helpful math assistant. Solve the problem.\n"
        "Put the final answer exactly once, in the form \\boxed{...}.\n\n"
        f"Question:\n{prompt}\n\nSolution:"
    )
    return template


