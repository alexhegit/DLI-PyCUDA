%%timeit
# Feel free to modify the 3 function calls in this cell
normalize(greyscales, out=normalized)
weigh(normalized, weights, out=weighted)
activate(weighted, out=SOLUTION)