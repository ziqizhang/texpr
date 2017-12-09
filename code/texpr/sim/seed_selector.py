import re

import numpy


# select 'size' terms from the file which contains a list of GS terms
def select_random_from_list(infile_term_list, size: int):
    regex = re.compile('[^a-zA-Z\-]')
    with open(infile_term_list, encoding='utf8') as f:
        output = []
        for line in f:
            processed = regex.sub(' ', line.strip())
            processed = re.sub(' +', ' ', processed)
            if len(processed) > 3:
                output.append(processed)
        return numpy.random.choice(output, size)
