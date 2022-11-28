from MultilayerPerceptron import MultilayerPerceptron
from Font import Font3, chrs, Font3_dict
import matplotlib.pyplot as plt

def propagate(optimus, font3):
    optimus.propagation(font3)
    exit = []
    for i in range(sum(optimus.npl[:-1]), optimus.nc):
        a = optimus.nds[i].v
        if a >= 0.5:
            a = 1
        else:
            a = 0
        exit.append(a)
    return exit

def get_bits_diff(font3, output):
    ret = 0
    for i in range(0, len(font3)):
        if font3[i] != output[i]:
            ret += 1
    return ret

def graph_error_by_bit(error):
    parameters = {'xtick.labelsize': 12,'ytick.labelsize': 12, 'axes.labelsize': 14}
    plt.rcParams.update(parameters)
    plt.figure(figsize=(7,5))
    plt.bar(chrs, error)
    plt.ylabel('Error by bit')
    plt.show()

def run_Ej1a2(optimus):
    bit_errors = []
    for h in range(0, len(Font3)):
        exit = propagate(optimus, Font3[h])
        bit_errors.append(get_bits_diff(Font3[h], exit))
    graph_error_by_bit(bit_errors)

def printResult(array):
    ret = []
    j = 0
    for i in array:
        if i == 1 or i == "1":
            ret.append("*")
        else:
            ret.append(" ")
        j = j + 1
        if j == 5:
            ret.append("\n")
            j=0
    return ret

def print_all_output(optimus):
    for h in range(0, len(Font3)):
        exit = propagate(optimus, Font3[h])
        print("\n-------------------------------------------------------------------------------------------\n")
        print("Character ->  \"", list(Font3_dict.keys())[list(Font3_dict.values()).index(Font3[h])], " \"\n")
        print("Input --> |\n", Font3[h], "|\n")
        print("Output -> |\n", exit, "|\n")

#def print_all_output(optimus):
#    for h in range(0, len(Font3)):
#        exit = propagate(optimus, Font3[h])
#       print("\n-------------------------------------------------------------------------------------------\n")
#        print("Character ->  \"", list(Font3_dict.keys())[list(Font3_dict.values()).index(Font3[h])], " \"\n")
#        print("Input --> |\n", *printResult(Font3[h]), "|\n")
#        print("Output -> |\n", *printResult(exit), "|\n")

beta = 0.5
learningRate = 0.001
e = len(Font3[0])
mlp = MultilayerPerceptron([e,15,2,15,e], learningRate, beta, Font3, Font3, 2)
mlp.run()
run_Ej1a2(mlp)
print_all_output(mlp)
