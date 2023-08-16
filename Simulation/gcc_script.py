import os

# Paths and compilation
file_compilation_strings = "compilation_strings.bat"
directory_executables = "./Compiled"
source_code_files = "../Autonomous_Edge_Pipeline/*.c"
gcc_warning_suppression_options = "2>nul"

# Static values
K = 2   # Number of clusters k-means
ITERATION = 50 # Max number of iterations k-means
N_TRAIN = 614
N_TEST = 154
N_TEST_USED = N_TEST

def generate_compilation_strings():
    CONF_THR_LIST = [.01,.05,.1,.2,.3,.4,.5,.6,.7,.8,.9]
    FILTERS = ['CONF','FIFO','RANDOM']

    output_file_names = []

    output_file = open(file_compilation_strings, "w")

    for N_NODES in range(1,6):
        N_TRAIN_USED = N_TRAIN / N_NODES
        for NODE_ID in N_NODES:
            for K_NEIGHBOR in range(1,11):
                for MEMORY_SIZE in range(10,N_TRAIN_USED,MEMORY_SIZE*2):
                    # TODO: Decide how to increase: is 10*2^i ok?
                    min_initial_thr = min(MEMORY_SIZE/2,50)
                    max_initial_thr = MEMORY_SIZE / 2
                    NODE_OFFSET = MEMORY_SIZE/N_NODES*(NODE_ID-1)
                    for CONFIDENCE in range(2): # False / True
                        for CONFIDENCE_THR in (CONF_THR_LIST if CONFIDENCE else [0]):
                            # The condition above should limit all the useless cases
                            for FILTER in FILTERS:
                                for ONE_SHOT in range(2): # False / True
                                    for INITIAL_THR in (range(min_initial_thr,max_initial_thr,50) if not ONE_SHOT else [0]):
                                        # TODO: Decide how to increase: is min+50*i ok?
                                        for UPDATE_THR in (range(5,100,UPDATE_THR*2) if not ONE_SHOT else [0]):
                                            # TODO: Decide how to increase: is 5*2^i ok?
                                            # TODO: Decide order, add everything to string, remember SIMULATION macro
                                            pass



    # for initial in INITIAL_THR:
    #     for update in UPDATE_THR:
    #         for size in MEMORY_SIZE:
    #             for iteration in ITERATION:
    #                 for confidence in CONFIDENCE_THR:
    #                     for k in K_NEIGHBOR:
    #                         for fil in FILTER:
    #                             for node in NODES:
    #                                 offset = node * size
    #                                 name_curr = f"{initial}_{update}_{size}_{iteration}_{confidence}_{k}_{fil}_{node}_{offset}"
    #                                 compilation_string = f"gcc "\
    #                                 f"-D INITIAL_THR={initial} "\
    #                                 f"-D UPDATE_THR={update} "\
    #                                 f"-D MEMORY_SIZE={size} "\
    #                                 f"-D ITERATION={iteration} "\
    #                                 f"-D CONFIDENCE_THR={confidence} "\
    #                                 f"-D K_NEIGHBOR={k} "\
    #                                 f"-D {fil} "\
    #                                 f"-D NODES={node} "\
    #                                 f"-D NODE_OFFSET={offset} "\
    #                                 f"-D SETTINGS=\"{name_curr}\" "\
    #                                 f"{gcc_warning_suppression_options} "\
    #                                 f"{source_code_files} "\
    #                                 f"-o {directory_executables}/{name_curr}"
    #                                 output_file.write(compilation_string + "\n")
    #                                 output_file_names.append(name_curr)

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
