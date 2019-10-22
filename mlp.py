import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')

# Activation functions
def Relu(x):
    return np.maximum(x, 0)

def BackwardRelu(x):
    xx = x.copy()
    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            if xx[i,j] > 0:
                xx[i,j] = 1
            else:
                xx[i,j] = 0
    return xx

def Sigmoid(x):
    return 1/(1+np.exp(-x))

def Softplus(x):
    return np.log(1+np.exp(x))

def Softmax(x):
    return np.exp(x)/np.array([np.sum(np.exp(x),axis=1)]).T


def DrawMap(model,x,y):
    plt.subplots(figsize=(12,8))
    data = np.random.uniform(-3, 3, (20000, 2))
    prediction = np.argmax(model.Predict(data), axis=1)
    c1 = []
    c2 = []
    c3 = []
    for i in range(len(data)):
        if prediction[i] == 0:
            c1.append(data[i])
        elif prediction[i] == 1:
            c2.append(data[i])
        else:
            c3.append(data[i])

    c1_x, c1_y = np.hsplit(np.array(c1),2)
    c2_x, c2_y = np.hsplit(np.array(c2),2)
    c3_x, c3_y = np.hsplit(np.array(c3),2)
    plt.scatter(c1_x, c1_y, c='r')
    plt.scatter(c2_x, c2_y, c='g')
    plt.scatter(c3_x, c3_y, c='b')

    indices_0 = [k for k in range(0, x.shape[0]) if y[k] == 0]
    indices_1 = [k for k in range(0, x.shape[0]) if y[k] == 1]
    indices_2 = [k for k in range(0, x.shape[0]) if y[k] == 2]

    plt.plot(x[indices_0, 0], x[indices_0,1], marker='o', linestyle='', ms=5, label='0')
    plt.plot(x[indices_1, 0], x[indices_1,1], marker='o', linestyle='', ms=5, label='1')
    plt.plot(x[indices_2, 0], x[indices_2,1], marker='o', linestyle='', ms=5, label='2')

    plt.show()


class HiddenLayer(object):
    def __init__(self, input_dim, output_dim, act_fun):
        self.act_fun = act_fun
        self.w = np.random.rand(output_dim, input_dim)-0.5
        self.b = np.random.rand(1, output_dim)-0.5

    def ForwardPropagation(self, inputs):
        self.inputs = inputs
        self.wxb = np.dot(inputs, np.transpose(self.w)) + self.b
        if self.act_fun == 'Relu':
            return Relu(self.wxb)
        elif self.act_fun == 'Softplus':
            return Softplus(self.wxb)
    
    def BackwardPropagation(self, dLdo, lr, alpha):
        if self.act_fun == 'Relu':
            dLdwxb = np.multiply(BackwardRelu(self.wxb), dLdo)
        elif self.act_fun == 'Softplus':
            dLdwxb = np.multiply(Sigmoid(self.wxb), dLdo)
        dLdw = np.dot(np.transpose(dLdwxb), self.inputs)
        dLdb = np.mean(dLdwxb, axis=0)
        dLdi = np.dot(dLdwxb, self.w)
        self.w -= lr*dLdw + alpha*2/self.inputs.shape[0]*self.w
        self.b -= lr*dLdb
        return dLdi
    
class OutputLayer(object):
    def __init__(self, input_dim, output_dim):
        self.w = np.random.rand(output_dim, input_dim)-0.5
        self.b = np.random.rand(1, output_dim)-0.5

    def ForwardPropagation(self, inputs):
        self.inputs = inputs
        return np.dot(inputs, np.transpose(self.w)) + self.b
    
    def BackwardPropagation(self, dLdo, lr, alpha):
        dLdw = np.dot(np.transpose(dLdo), self.inputs)
        dLdb = np.mean(dLdo, axis=0)
        dLdi = np.dot(dLdo, self.w)
        self.w -= lr*dLdw + alpha*2/self.inputs.shape[0]*self.w
        self.b -= lr*dLdb
        return dLdi
    
class LossLayer(object):
    def ForwardPropagation(self, inputs, label):
        loss = - np.sum(np.multiply(label,np.log(Softmax(inputs))))/label.shape[0]
        return loss
    
    def BackwardPropagation(self, inputs, label):
        return Softmax(inputs) - label


class MLP(object):
    def __init__(self):
        self.layer_list = []
        
    def AddLayer(self, layer_type, input_dim, output_dim, act_fun):
        if layer_type == 'Hidden':
            self.layer_list.append(HiddenLayer(input_dim, output_dim, act_fun))
            return
        elif layer_type == 'Output':
            self.layer_list.append(OutputLayer(input_dim, output_dim))
            return
        elif layer_type == 'Loss':
            self.layer_list.append(LossLayer())
            return
        else:
            print('Invalid layer name')
            return
        
    def Forward(self, inputs):
        output = inputs.copy()
        reg = 0
        for i in range(len(self.layer_list)-1):
            output = self.layer_list[i].ForwardPropagation(output)
            reg += np.linalg.norm(self.layer_list[i].w)
        return output, reg
    
    def Backward(self, z, label, alpha):
        grad = self.layer_list[-1].BackwardPropagation(z, label)
        for i in range(len(self.layer_list)-2,-1,-1):
            grad = self.layer_list[i].BackwardPropagation(grad, self.lr, alpha)
    
    def Predict(self, inputs):
        output, reg = self.Forward(inputs)
        return Softmax(output)
    
    def Train(self, x, y, num_epoch, batch_size, lr, alpha):
        self.lr = lr
        input_dim = x.shape[1]
        output_dim = y.shape[1]
        num_batch = int(len(x)/batch_size)
        lit = list(range(num_epoch))
        loss_list = []
        data = np.concatenate((x,y),axis=1)
        for i in range(num_epoch):
            if i == int(num_epoch/2):
                lr = lr/10
            data_index = list(range(len(x)))           
            batch_index = []
            loss = 0
            for j in range(num_batch):
                batch_index.append(random.sample(data_index, batch_size))
                for k in range(len(batch_index[j])):
                    data_index.remove(batch_index[j][k])
            for index in batch_index:
                training = data[index]
                training_split = np.hsplit(training, input_dim + output_dim)
                inputs = np.concatenate(([training_split[i] for i in range(input_dim)]),axis=1)
                label = np.concatenate(([training_split[i] for i in range(input_dim, input_dim + output_dim)]),axis=1)
                z, reg = self.Forward(inputs)
                loss += self.layer_list[-1].ForwardPropagation(z, label) + alpha*reg
                self.Backward(z, label, alpha)
              
            
            loss_list.append(loss/num_batch)
            print(i,"/",num_epoch,",Loss=", loss/num_batch)
            
        plt.subplots()      
        plt.plot(lit, loss_list) 
        plt.title('Loss')
        plt.draw()
