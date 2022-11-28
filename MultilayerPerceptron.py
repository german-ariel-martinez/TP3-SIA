from Perceptron import Perceptron
from Font import Font3
import numpy as np

class MultilayerPerceptron:

    def __init__(self, npl, lr, b, st, eo, lli):
        # Nodes Per Layer (incluye la entrada y la salida)
        self.npl = npl
        # Cantidad total de nodos
        self.nc  = sum(npl)
        # Learning Rate
        self.lr  = lr
        # Beta
        self.b   = b
        # Stimulus
        self.st  = st
        # Expected Output
        self.eo  = eo
        # Synaptic Efficiencies Matrix
        self.sem = self.sem_generator()
        # Synaptic Efficiencies Initializer
        self.sem_builder()
        # Node array
        self.nds = self.nds_generator()
        # Latent Layer Index
        self.lli = lli
        # Output Layer
        self.ol  = len(npl) - 1

    def g(self, x):
        return np.tanh(self.b * x)

    def g_dx_dt(self, x):
        return self.b * (1 - (self.g(x)) ** 2)

    def sem_generator(self):
        # La matriz sera de NxN con N = todos los nodos del sistema
        matrix = []
        for _ in range(0, self.nc):
            row = [None for _ in range(0, self.nc)]
            matrix.append(row)
        return matrix

    def sem_builder(self):
        #  [2,1,2]
        #    1 2 3 4 5
        # 1  0 0 1 0 0
        # 2    0 1 0 0
        # 3      0 1 1
        # 4        0 0
        # 5          0
        # Aca tenemos que poner los Ws de cada conexion sinaptica
        # nos va a quedar una matriz espejada. La diagonal superior
        # nos va a dar los pesos de izq a der y la inferior de der a 
        # izq.
        pivot = 0
        for idx, nodes in enumerate(self.npl[:-1]):
            for i in range(0, nodes):
                for j in range(0, self.npl[idx+1]):
                    value = np.random.uniform(low=-1, high=1)
                    self.sem[i+pivot][pivot+nodes+j] = value
                    self.sem[pivot+nodes+j][i+pivot] = value
            pivot += nodes

    def sem_update(self):
        pivot = 0
        for idx, nodes in enumerate(self.npl[:-1]):
            for i in range(0, nodes):
                for j in range(0, self.npl[idx+1]):
                    self.sem[i+pivot][pivot+nodes+j] += self.lr * self.nds[pivot+j].d * self.nds[i].v
                    self.sem[pivot+nodes+j][i+pivot] += self.lr * self.nds[pivot+j].d * self.nds[i].v
            pivot += nodes

    def sem_print(self):
        # Por propositos de visualizacion
        for i in range(0, self.nc):
            for j in range(0, self.nc):
                print(str(self.sem[i][j]) + " ", end='')
            print('\n')

    def nds_generator(self):
        # Genera todos los nodos del sistema
        # y los indexa
        nds = []
        for l, nodes in enumerate(self.npl):
            for _ in range(0, nodes):
                nds.append(Perceptron(l))
        return nds

    def nds_layer_print(self, layer):
        # Imprime de forma legible los nodos de
        # una capa especifica
        pivot = sum(self.npl[:layer])
        for n in range(pivot, pivot+self.npl[layer]):
            print(self.nds[n])

    def nds_st_initializer(self, st):
        # Inicializa las entradas con los estimulos
        for i in range(0, self.npl[0]):
            self.nds[i].v = st[i]

    def run(self):
        # Corte por iteraciones
        max_i = 20000
        i     = 0
        # Corte por error
        error  = 1
        ac_err = 1e-5
        # Optimizacion
        error_history = []
        it = []
        e_min = 1000000
        w_min = []
        # Run
        while i < max_i and error > ac_err:
            error = 0
            for st in self.st:
                error = 0
                self.propagation(st)
                self.calculate_exit_delta(st)
                self.backpropagation()
                self.sem_update()
                error += self.calculate_error(st)
            if (i % 1000 == 0):
                print(i)
            i += 1
    



    def get_layer_index(self,layer):
        return sum(self.npl[:layer])

    def get_layer_nodes(self,layer):
        idx = self.get_layer_index(layer)
        return self.nds[idx: idx+self.npl[layer]]
        
    # 3 estimulos y estructura [3, 2, 1, 2, 3] 3 outputs => hay 7 capas. La matriz es de (7*5)x(7*5)
    #    1 2 3 4 5 6 7 8 9 10 11 
    # 1  0 0 0 1 1 0 0 0 0 0 0
    # 2  0 0 0 1 1 0 0 0 0 0 0
    # 3  0 0 0 1 1 0 0 0 0 0 0
    # 4  1 1 1 0 0 1 0 0 0 0 0
    # 5  1 1 1 0 0 1 0 0 0 0 0
    # 6  0 0 0 1 1 0 1 1 0 0 0
    # 7  0 0 0 0 0 1 0 0 1 1 1
    # 8  0 0 0 0 0 1 0 0 1 1 1
    # 9  0 0 0 0 0 0 1 1 0 0 0
    # 10 0 0 0 0 0 0 1 1 0 0 0
    # 11 0 0 0 0 0 0 1 1 0 0 0



    def propagation(self, st):
        # Propaga el estimulo hasta la entrada
        self.nds_st_initializer(st)
        # Aprovechamos que la matriz es espejada para
        # saltearnos iteraciones de mas
        pivot = self.npl[0]
        for npl in self.npl[1:]:
            for j in range(pivot, pivot+npl):
                h = 0
                #print(" j = " + str(j+1))
                for i in range(pivot-self.npl[self.nds[j].l-1], pivot):
                    #print(" i = " + str(i+1) + " *")
                    h += self.sem[i][j] * self.nds[i].v
                self.nds[j].h = h
                self.nds[j].v = self.g(h)
            pivot += npl

    def calculate_exit_delta(self, st):
        # Calculamos el delta para la capa de salida
        pivot = sum(self.npl[:-1])
        for n in range(pivot, self.nc):
            self.nds[n].d = self.g_dx_dt(self.nds[n].h) * (st[n-pivot] - self.nds[n].v)

    def backpropagation(self):
        # Indices de los nodos inteiores en forma reversa
        for n in reversed(range(self.npl[0], self.nc - self.npl[self.ol])):
            sm = 0
            #print(" j = " + str(n+1))
            pivot = sum(self.npl[:-(self.ol-self.nds[n].l)])
            for j in range(pivot,pivot+self.npl[self.nds[n].l+1]):
                #print(" i = " + str(j+1) + " *")
                sm += self.sem[j][n] * self.nds[j].d
            self.nds[n].d = self.g_dx_dt(self.nds[n].h) * sm

    def calculate_error(self, st):
        err = 0
        pivot = sum(self.npl[:-1])
        for i in range(pivot, self.nc):
            err += (st[i-pivot] - self.nds[i].v)**2
        return err