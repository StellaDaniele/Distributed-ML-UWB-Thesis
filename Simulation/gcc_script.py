import os

file_compilation_strings = "compilation_strings.bat"
directory_executables = "./Compiled"
source_code_files = "../Autonomous_Edge_Pipeline/*.c"
gcc_warning_suppression_options = "2>nul"

def generate_compilation_strings():
    INITIAL_THR = range(1)     # 50
    UPDATE_THR = range(1)     # 100
    MEMORY_SIZE = range(1)    # 200

    ITERATION = range(1)       # 50
    CONFIDENCE_THR = range(1)        # 0.9

    K_NEIGHBOR = range(1)       # 5

    FILTER = ['CONF','FIFO','RANDOM']

    NODES = range(1)

    output_file_names = []

    output_file = open(file_compilation_strings, "w")

    for initial in INITIAL_THR:
        for update in UPDATE_THR:
            for size in MEMORY_SIZE:
                for iteration in ITERATION:
                    for confidence in CONFIDENCE_THR:
                        for k in K_NEIGHBOR:
                            for fil in FILTER:
                                for node in NODES:
                                    offset = node * size
                                    name_curr = f"{initial}_{update}_{size}_{iteration}_{confidence}_{k}_{fil}_{node}_{offset}"
                                    compilation_string = f"gcc "\
                                    f"-D INITIAL_THR={initial} "\
                                    f"-D UPDATE_THR={update} "\
                                    f"-D MEMORY_SIZE={size} "\
                                    f"-D ITERATION={iteration} "\
                                    f"-D CONFIDENCE_THR={confidence} "\
                                    f"-D K_NEIGHBOR={k} "\
                                    f"-D {fil} "\
                                    f"-D NODES={node} "\
                                    f"-D NODE_OFFSET={offset} "\
                                    f"-D SETTINGS=\"{name_curr}\" "\
                                    f"{gcc_warning_suppression_options} "\
                                    f"{source_code_files} "\
                                    f"-o {directory_executables}/{name_curr}"
                                    output_file.write(compilation_string + "\n")
                                    output_file_names.append(name_curr)

    return output_file_names

if __name__ == "__main__":
    if not os.path.exists(directory_executables):
        os.makedirs(directory_executables)

    # Gcc compilation string generation
    output_file_names = generate_compilation_strings()
    for file_name in output_file_names:
        print(file_name)

    # Gcc compilation execution
    os.system(file_compilation_strings)

    # Execution of all compiled programs
    for executable_name in output_file_names:
        os.system(directory_executables + "/" + executable_name + ".exe")

    # Reading and concatenating all the logs
    # for log_name in output_file_names:
    #     log_file = open(directory_executables + "/" + "log_" + log_name + ".txt", "r")
