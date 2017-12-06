import numpy

#select 'size' terms from the file which contains a list of GS terms
def select_random_from_list(infile_term_list, size:int):
    with open(infile_term_list, encoding='utf8') as f:
        lines = f.read().splitlines()
        return numpy.random.choice(lines,size)



