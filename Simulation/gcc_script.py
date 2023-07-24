def generate_compilation_strings():
    INITIAL_THR = range(1)     # 50
    UPDATE_THR = range(1)     # 100
    MEMORY_SIZE = range(1)    # 200

    ITERATION = range(1)       # 50
    CONFIDENCE_THR = range(1)        # 0.9

    K_NEIGHBOR = range(1)       # 5

    FILTER = ['CONF','FIFO','RANDOM']

    NODES = range(1)
    NODE_OFFSET = range(1)

    compilation_strings = []

    for initial in INITIAL_THR:
        for update in UPDATE_THR:
            for size in MEMORY_SIZE:
                for iteration in ITERATION:
                    for confidence in CONFIDENCE_THR:
                        for k in K_NEIGHBOR:
                            for fil in FILTER:
                                for node in NODES:
                                    for offset in NODE_OFFSET:
                                        compilation_string = f"gcc -D INITIAL_THR={initial} -D UPDATE_THR={update} -D MEMORY_SIZE={size} -D ITERATION={iteration} -D CONFIDENCE_THR={confidence} -D K_NEIGHBOR={k} -D FILTER={fil} -D NODES={node} -D NODE_OFFSET={offset} *.c -o output_{initial}_{update}_{size}_{iteration}_{confidence}_{k}_{fil}_{node}_{offset}"
                                        compilation_strings.append(compilation_string)

    return compilation_strings

if __name__ == "__main__":
    compilation_strings = generate_compilation_strings()

    with open("compilation_strings.txt", "w") as file:
        for string in compilation_strings:
            file.write(string + "\n")
