
def split_into_parts(text):
    result = [text]

    if text:
        ##  check if text looks like a tuple of list---e.g "(a,b)"  or "[a,b]"
        if (len(text) >= 2 and text[0] in "([" and text[-1] in ")]" and "," in text[1:-1]):
            ## split on comma
            items = [i.strip() for i in text[1:-1].split(",")]
            if all(items):
                result = items 

    else:
        ## text is empty, so return the empty list
        result = []

    return result 