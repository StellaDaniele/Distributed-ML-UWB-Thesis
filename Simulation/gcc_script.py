import os

# Paths and script settings
directory_executables = "./Compiled"
directory_compilation_strings = "./Compilation_Strings"
source_code_files = "../../Autonomous_Edge_Pipeline/*.c"
gcc_warning_suppression_options = "" # "2>nul"
file_name_compilation = "compilation_strings"
file_name_compilation_prefix = directory_compilation_strings + "/" + file_name_compilation
extension_file_compilation = ".bat"
batch_dimension_compilation_files = 1000

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
    compilation_files_counter = 0
    current_strings_on_file = 0
    total_compilation_strings_generated = 0
    output_file = open(file_name_compilation_prefix+str(compilation_files_counter)+extension_file_compilation, "w")
    compilation_files_counter += 1

    for N_NODES in range(1,6):
        N_TRAIN_USED = N_TRAIN // N_NODES
        for NODE_ID in range(1,N_NODES+1):
            for K_NEIGHBOR in range(1,11):
                MEMORY_SIZE = 10 # Initial value to decide
                while MEMORY_SIZE <= N_TRAIN_USED:
                    min_initial_thr = min(MEMORY_SIZE//2,50)
                    max_initial_thr = MEMORY_SIZE // 2
                    NODE_OFFSET = MEMORY_SIZE//N_NODES*(NODE_ID-1)
                    for CONFIDENCE in range(2): # False / True
                        for CONFIDENCE_THR in (CONF_THR_LIST if CONFIDENCE else [0]):
                            # The condition above should limit all the useless cases
                            for FILTER in FILTERS:
                                for ONE_SHOT in range(2): # False / True
                                    for INITIAL_THR in (range(min_initial_thr,max_initial_thr,50) if not ONE_SHOT else [0]):
                                        # TODO: Decide how to increase: is min+50*i ok?
                                        update_thr_values = [0]
                                        if not ONE_SHOT:
                                            min_update_thr_value = 5
                                            max_update_thr_value = 100
                                            current_value = min_initial_thr
                                            while current_value <= max_update_thr_value and INITIAL_THR+current_value <=MEMORY_SIZE:
                                                update_thr_values.append(current_value)
                                                current_value *= 2
                                        for UPDATE_THR in update_thr_values:
                                            # TODO: Decide how to increase: is 5*2^i ok?
                                            name_curr = f"{N_NODES}_{K_NEIGHBOR}_{MEMORY_SIZE}_{NODE_OFFSET}_{CONFIDENCE}_{CONFIDENCE_THR}_{FILTER}_{ONE_SHOT}_{INITIAL_THR}_{UPDATE_THR}_{K}_{ITERATION}_{N_TRAIN}_{N_TRAIN_USED}_{N_TEST}_{N_TEST_USED}_{NODE_ID}"
                                            # NODE_ID is last so that we can use sorting to find all the nodes given the settings
                                            compilation_string = f"gcc -D SIMULATION "\
                                            f"-D N_NODES={N_NODES} "\
                                            f"-D NODE_ID={NODE_ID} "\
                                            f"-D K_NEIGHBOR={K_NEIGHBOR} "\
                                            f"-D MEMORY_SIZE={MEMORY_SIZE} "\
                                            f"-D NODE_OFFSET={NODE_OFFSET} "\
                                            f"-D CONFIDENCE={CONFIDENCE} "\
                                            f"-D CONFIDENCE_THR={CONFIDENCE_THR} "\
                                            f"-D {FILTER} "\
                                            f"-D ONE_SHOT={ONE_SHOT} "\
                                            f"-D INITIAL_THR={INITIAL_THR} "\
                                            f"-D UPDATE_THR={UPDATE_THR} "\
                                            f"-D K={K} "\
                                            f"-D ITERATION={ITERATION} "\
                                            f"-D N_TRAIN={N_TRAIN} "\
                                            f"-D N_TRAIN_USED={N_TRAIN_USED} "\
                                            f"-D N_TEST={N_TEST} "\
                                            f"-D N_TEST_USED={N_TEST_USED} "\
                                            f"-D SETTINGS=\\\"{name_curr}\\\" "\
                                            f"{gcc_warning_suppression_options} "\
                                            f"{source_code_files} "\
                                            f"-o {directory_executables}/{name_curr}"
                                            output_file.write(compilation_string + "\n")
                                            current_strings_on_file += 1
                                            total_compilation_strings_generated += 1
                                            if(current_strings_on_file % batch_dimension_compilation_files == 0):
                                                output_file.close()
                                                output_file = open(file_name_compilation_prefix+str(compilation_files_counter)+extension_file_compilation, "w")
                                                compilation_files_counter += 1
                                                current_strings_on_file = 0
                                            output_file_names.append(name_curr)
                    MEMORY_SIZE *= 2 # TODO: Decide how to increase: is 10*2^i ok?
    return output_file_names, compilation_files_counter, total_compilation_strings_generated

if __name__ == "__main__":
    if not os.path.exists(directory_executables):
        os.makedirs(directory_executables)

    # Gcc compilation string generation
    output_file_names, compilation_files_counter, total_compilation_strings_generated = generate_compilation_strings()
    # for file_name in output_file_names:
    #     print(file_name)
    print("# Files: "+str(compilation_files_counter))
    print("# Compilation strings: "+str(total_compilation_strings_generated))
    # # Gcc compilation execution
    # os.system(file_compilation_strings)

    # # Execution of all compiled programs
    # for executable_name in output_file_names:
    #     os.system(directory_executables + "/" + executable_name + ".exe")

    # Reading and concatenating all the logs
    # for log_name in output_file_names:
    #     log_file = open(directory_executables + "/" + "log_" + log_name + ".txt", "r")
