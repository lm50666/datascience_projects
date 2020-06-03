import math
import numpy as np
import random
def sigmoid(t):
    return 1/(1+math.exp(-t))

def neuron_output(weights,inputs):
    return sigmoid(np.dot(weights,inputs))

def feed_forward(neural_network,input_vector):
    outputs=[]
    for layer in neural_network:
        input_with_bias=input_vector +[1]
        output=[neuron_output(neuron,input_with_bias) for neuron in layer]
        outputs.append(output)
        input_vector=output
    return outputs
def backpropagate(network,input_vector,targets):

    #produciranje svih outputova za sve neurone pomocu feed_forward
    hidden_outputs,outputs=feed_forward(network,input_vector)
    #izracunvanje razlike outputa i targeta
    output_deltas=[output*(1-output)*(output-target) for output,target in zip(outputs,targets)]

    #adjusting weights for output_neurons

    for i,output_neuron in enumerate(network[-1]):
        for j,hidden_output in enumerate(hidden_outputs+[1]):
            output_neuron[j]-=output_deltas[i]*hidden_output

    #back-propagate
    hidden_deltas=[hidden_output*(1-hidden_output)*np.dot(output_deltas,[n[i] for n in network[-1]])
                   for i,hidden_output in enumerate(hidden_outputs)]

    #adjusting weights for hidden layer
    for i,hidden_neuron in enumerate(network[0]):
        for j,input in enumerate(input_vector+[1]):
            hidden_neuron[j]-=hidden_deltas[i]*input


def predict(network,input):
    return feed_forward(network,input)[-1]

global output_layer

def main():
    digit0=[1,1,1,1,1,1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,1,1,1,1,1]
    digit1=[0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0]
    digit2=[1,1,1,1,1,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,1,1,1,1,1]
    digit3=[1,1,1,1,1,0,0,0,0,1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,1]
    digit4=[1,0,0,0,1,1,0,0,0,1,1,1,1,1,1,0,0,0,0,1,0,0,0,0,1]
    digit5=[1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,0,0,0,0,1,1,1,1,1,1]
    digit6=[1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,1,0,0,0,1,1,1,1,1,1]
    digit7=[1,1,1,1,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1]
    digit8=[1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,0,0,0,1,1,1,1,1,1]
    digit9=[1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,1]
    inputs=[digit0,digit1,digit2,digit3,digit4,digit5,digit6,digit7,digit8,digit9]
    targets=[[1 if i==j else 0 for i in range(10)] for j in range(10)]
    random.seed(0)
    input_size=25
    num_hidden=5
    output_size=10
    hidden_layer = [[random.random() for _ in range(input_size + 1)]
                    for _ in range(num_hidden)]
    output_layer=[[random.random() for _ in range (num_hidden+1)]
    for _ in range(output_size)]
    network=[hidden_layer, output_layer]

    for _ in range(1000):
        for input_vector,target_vector in zip(inputs,targets):
            backpropagate(network,input_vector,target_vector)
    a=predict(network,inputs[3])
    print(a)
main()
