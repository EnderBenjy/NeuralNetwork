from network import *

import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

final = []
'''
for i in range(6):
    for j in range(5):
        results = []
        indice = [784] + [(j+1)*10]*i + [10]
        net = Network(indice)
        print(indice)
        print(f'\n Generation {i+j}: ')
        net.SGD(training_data, 10, 10, 3.0, test_data=test_data)
        final.append(results)
'''
'''
net = Network([784,32,32,10])
net.SGD(training_data, 20, 10, 3.0, test_data=test_data)
print(f'\n \n \n \n Resultats: {final}')
'''

nb_neurones = [8,16,32]
nb_layers = [0,1,2,5]

net3 = Network(784,10)
net4 = Network(784,8,10)
net5 = Network(784,16,10)
net6 = Network(784,32,10)
net7 = Network(784,8,8,10)
net8 = Network(784,16,16,10)
net9 = Network(784,32,32,10)
net10 = Network(784,8,8,8,8,8,10)
net11 = Network(784,16,16,16,16,16,10)
net12 = Network(784,32,32,32,32,32,10)

reseaux = [net3, net4, net5, net6, net7, net8, net9, net10, net11, net12]

for network in reseaux:
    network.SGD(training_data, 10, 10, 3.0, test_data=test_data)
