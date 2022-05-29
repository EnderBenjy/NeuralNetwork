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
for i in range(5):
    print(f"net1{i} = Network([784, {i*10+10}, 10])")

for i in range(5):
    print(f"net2{i} = Network([784, {i*10+10}, {i*10+10}, 10])")

for i in range(5):
    print(f"net3{i} = Network([784, {i*10+10}, {i*10+10}, {i*10+10}, 10])")
