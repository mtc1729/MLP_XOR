import numpy as nm
import math
import random
import matplotlib.pyplot as plt
from numpy import sin, cos, pi
from scipy.optimize import leastsq
INPUT_NODES=2
OUTPUT_NODES=1
HIDDEN_NODES=2
LEARNING_RATE=0.15
MAX_ITERATIONS=1020
VALUE=200
epsilon=0.000001
W=500

    
class neural_network:

    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        self.momentum1=nm.zeros((4,3))
        self.momentum2=nm.zeros((4,2))
        self.gradient1=nm.zeros((4,3))
        self.gradient2=nm.zeros((4,2))

        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        
        self.learning_rate = learning_rate

        # set up the arrays
        self.values = nm.zeros(self.input_nodes+self.output_nodes+1)
        self.val_in = nm.zeros(self.input_nodes+1)
        self.thresholds = nm.zeros(self.input_nodes+2)



        
        self.weights1 = nm.zeros((self.input_nodes+1, self.hidden_nodes+1))
        self.weights2 = nm.zeros((self.hidden_nodes+1, self.output_nodes+1))

        # set random seed! this is so we can experiment consistently
        random.seed(10000)

        # set initial random values for weights and thresholds
        # this is a strictly upper triangular matrix as there is no feedback
        # loop and there inputs do not affect other inputs

        for i in range(0, self.input_nodes+1):
            for j in range(1, self.hidden_nodes+1):
                self.weights1[i][j] = 0.4*random.random()+0.2
                
    
        for i in range(0, self.hidden_nodes+1):
            for j in range(1, self.output_nodes+1):
                self.weights2[i][j] = 0.4*random.random()+0.2
    
        
    
    def init_arr(self):
            
        self.momentum1=nm.zeros((4,3))
        self.momentum2=nm.zeros((4,2))
        
        
        
    def process(self,bias,x):
        # update the hidden nodes
                for i in range(1, self.input_nodes+1):
                    self.val_in[i]=x[i-1]  
                    
                for i in range(1, self.hidden_nodes+1):
                    # sum weighted input nodes for each hidden node, compare threshold, apply sigmoid
                     W_i = self.weights1[0][i] * bias
                     for j in range(1,self.input_nodes+1):
                        W_i += self.weights1[j][i] * x[j-1]
               #      W_i -= self.thresholds[i]   
                     self.values[i] = 1 / (1 + math.exp(-W_i))

        # update the output nodes

            # sum weighted hidden nodes for each output node, compare threshold, apply sigmoid
                W_3 = self.weights2[0][1] * bias
                for j in range(1,self.hidden_nodes+1):
                   W_3 += self.weights2[j][1] * self.values[j]
             #   W_3 -= self.thresholds[3]
                self.values[3] = 1 / (1 + math.exp(-W_3))

    def calcError_sgd(self,x):
            sumOfSquaredErrors = 0.0
            delta1 = nm.zeros(self.input_nodes+1)
        # we only look at the output nodes for error calculation
      
            error = x[2] - self.values[3]
            #print error


            outputErrorGradient = (self.values[3]) * error *(1-self.values[3])
            #print outputErrorGradient
            delta = self.learning_rate * outputErrorGradient
                        #print delta
            self.weights2[0][1] += delta   
            
                        
            # now update the weights and thresholds
            for j in range(1, self.hidden_nodes+1):
            

                # first update for the hidden nodes to output nodes (1 layer)
                        delta1[j] = self.learning_rate * self.values[j] * outputErrorGradient
                        #print delta
                        
                        hiddenErrorGradient = (self.values[j])*(1-self.values[j]) * outputErrorGradient * self.weights2[j][1]
                        
                # and then update for the input nodes to hidden nodes
                        delta = self.learning_rate * hiddenErrorGradient
                        self.weights1[0][j] += delta

                        for k in range(1,self.input_nodes+1):
                            delta = self.learning_rate * self.val_in[k] * hiddenErrorGradient
                            self.weights1[k][j] += delta
                #        delta = self.learning_rate * -1 * hiddenErrorGradient
                #print delta
                 #       self.thresholds[j] += delta
            for j in range(1, self.hidden_nodes+1):
                self.weights2[j][1] += delta1[j]   
            # update the thresholds for the output node(s)
            #delta = self.learning_rate * -1 * outputErrorGradient
            #self.thresholds[3] += delta

                       

                # update the thresholds for the hidden nodes
            return math.pow(error,2)

    def calcError_momentum(self,x,gamma):
            sumOfSquaredErrors = 0.0
            
            delta1 = nm.zeros(self.input_nodes+1)
        # we only look at the output nodes for error calculation
      
            error = x[2] - self.values[3]
            #print error


            outputErrorGradient = (self.values[3]) * error *(1-self.values[3])*-1
            #print outputErrorGradient
            delta = self.learning_rate * outputErrorGradient
            self.momentum2[0][1]=(gamma*self.momentum2[0][1])+delta
                        #print delta
            self.weights2[0][1] -=self.momentum2[0][1]   
            
                        
            # now update the weights and thresholds
            for j in range(1, self.hidden_nodes+1):
            

                # first update for the hidden nodes to output nodes (1 layer)
                        delta1[j] = self.learning_rate * self.values[j] * outputErrorGradient
                        #print delta
                        
                        hiddenErrorGradient = (self.values[j])*(1-self.values[j]) * outputErrorGradient * self.weights2[j][1]
                        
                # and then update for the input nodes to hidden nodes
                        delta = self.learning_rate * hiddenErrorGradient
                        self.momentum1[0][j]=(gamma*self.momentum1[0][j])+delta
                        self.weights1[0][j] -= self.momentum1[0][j]
                        
                        

                        for k in range(1,self.input_nodes+1):
                            delta = self.learning_rate * self.val_in[k] * hiddenErrorGradient
                            self.momentum1[k][j]=(gamma*self.momentum1[k][j])+delta
                            self.weights1[k][j] -=self.momentum1[k][j]
                #        delta = self.learning_rate * -1 * hiddenErrorGradient
                #print delta
                 #       self.thresholds[j] += delta
            for j in range(1, self.hidden_nodes+1):
                self.momentum2[j][1]=(gamma*self.momentum2[j][1])+delta1[j]
                self.weights2[j][1] -= self.momentum2[j][1]
            # update the thresholds for the output node(s)
            #delta = self.learning_rate * -1 * outputErrorGradient
            #self.thresholds[3] += delta

                       

                # update the thresholds for the hidden nodes
            return math.pow(error,2)

    def calcError_nag(self,x,gamma):
            sumOfSquaredErrors = 0.0
            
            delta1 = nm.zeros(self.input_nodes+1)
        # we only look at the output nodes for error calculation
            error1=x[2] - self.values[3]
            error=0
            for j in range(0,3):
               error = error+ x[2] - (self.values[3] - gamma*self.momentum2[j][1])
            #print error

            outputErrorGradient = (self.values[3]) * error *(1-self.values[3])*-1
            #print outputErrorGradient
            delta = self.learning_rate * outputErrorGradient 
            self.momentum2[0][1]=(gamma*self.momentum2[0][1])+delta
                        #print delta
            self.weights2[0][1] -=self.momentum2[0][1]   
            
                        
            # now update the weights and thresholds
            for j in range(1, self.hidden_nodes+1):
            

                # first update for the hidden nodes to output nodes (1 layer)
                        delta1[j] = self.learning_rate * outputErrorGradient * (self.values[j]+gamma*self.momentum2[j][1])
                        #print delta
                        
                        hiddenErrorGradient = (self.values[j])*(1-self.values[j]) * outputErrorGradient * (self.weights2[j][1] - (gamma*self.momentum1[j][1]))
                        
                # and then update for the input nodes to hidden nodes
                        delta = self.learning_rate * hiddenErrorGradient 
                        self.momentum1[0][j]=(gamma*self.momentum1[0][j])+delta
                        self.weights1[0][j] -= self.momentum1[0][j]
                        
                        

                        for k in range(1,self.input_nodes+1):
                            delta = self.learning_rate *  hiddenErrorGradient * (self.val_in[k] + gamma*self.momentum1[k][j])
                            self.momentum1[k][j]=(gamma*self.momentum1[k][j])+delta
                            self.weights1[k][j] -=self.momentum1[k][j]
                #        delta = self.learning_rate * -1 * hiddenErrorGradient
                #print delta
                 #       self.thresholds[j] += delta
            for j in range(1, self.hidden_nodes+1):
                self.momentum2[j][1]=(gamma*self.momentum2[j][1])+delta1[j]
                self.weights2[j][1] -= self.momentum2[j][1]
            # update the thresholds for the output node(s)
            #delta = self.learning_rate * -1 * outputErrorGradient
            #self.thresholds[3] += delta

                       

                # update the thresholds for the hidden nodes
            return math.pow(error1,2)
    def calcError_adagrad(self,x):
            sumOfSquaredErrors = 0.0
            
            delta1 = nm.zeros(self.input_nodes+1)
        # we only look at the output nodes for error calculation
      
            error = x[2] - self.values[3]
            #print error


            outputErrorGradient = (self.values[3]) * error *(1-self.values[3])*-1
            #print outputErrorGradient
            delta = self.learning_rate * outputErrorGradient
           
            self.momentum2[0][1]=self.momentum2[0][1]+(1 * outputErrorGradient)**2
                        #print delta
            self.weights2[0][1] -=delta/((self.momentum2[0][1]**0.5)+epsilon)   
            
                        
            # now update the weights and thresholds
            for j in range(1, self.hidden_nodes+1):
            

                # first update for the hidden nodes to output nodes (1 layer)
                        delta1[j] = self.learning_rate * self.values[j] * outputErrorGradient
                        #print delta
                        
                        hiddenErrorGradient = (self.values[j])*(1-self.values[j]) * outputErrorGradient * self.weights2[j][1]
                        
                # and then update for the input nodes to hidden nodes
                        delta = self.learning_rate * hiddenErrorGradient
                        self.momentum1[0][j]=self.momentum1[0][j]+(1 * hiddenErrorGradient)**2
                        #print delta
                        self.weights1[0][j] -=delta/((self.momentum1[0][1]**0.5)+epsilon)   
                        
                        

                        for k in range(1,self.input_nodes+1):
                            delta = self.learning_rate * self.val_in[k] * hiddenErrorGradient
                            self.momentum1[k][j]=self.momentum1[k][j]+(self.val_in[k] * hiddenErrorGradient)**2
                            self.weights1[k][j] -=delta/((self.momentum1[k][1]**0.5)+epsilon)   
                #        delta = self.learning_rate * -1 * hiddenErrorGradient
                #print delta
                 #       self.thresholds[j] += delta
            for j in range(1, self.hidden_nodes+1):
                self.momentum2[j][1]=self.momentum2[j][1]+(self.values[j] * outputErrorGradient)**2
                self.weights2[j][1] -= delta1[j]/((self.momentum2[j][1]**0.5)+epsilon) 
            # update the thresholds for the output node(s)
            #delta = self.learning_rate * -1 * outputErrorGradient
            #self.thresholds[3] += delta

                       

                # update the thresholds for the hidden nodes
            return math.pow(error,2)
    
    
            
    def calcError_rmsprop(self,x,eta):
            sumOfSquaredErrors = 0.0
            
            delta1 = nm.zeros(self.input_nodes+1)
        # we only look at the output nodes for error calculation
      
            error = x[2] - self.values[3]
            #print error


            outputErrorGradient = (self.values[3]) * error *(1-self.values[3])*-1
            #print outputErrorGradient
            delta = self.learning_rate * outputErrorGradient
           
            self.momentum2[0][1]=eta*self.momentum2[0][1]+((1-eta)*(1 * outputErrorGradient)**2)
                        #print delta
            self.weights2[0][1] -=delta/((self.momentum2[0][1]**0.5)+epsilon)   
            
                        
            # now update the weights and thresholds
            for j in range(1, self.hidden_nodes+1):
            

                # first update for the hidden nodes to output nodes (1 layer)
                        delta1[j] = self.learning_rate * self.values[j] * outputErrorGradient
                        #print delta
                        
                        hiddenErrorGradient = (self.values[j])*(1-self.values[j]) * outputErrorGradient * self.weights2[j][1]
                        
                # and then update for the input nodes to hidden nodes
                        delta = self.learning_rate * hiddenErrorGradient
                        self.momentum1[0][j]=eta*self.momentum1[0][j]+((1-eta)*(1 * hiddenErrorGradient)**2)
                        #print delta
                        self.weights1[0][j] -=delta/((self.momentum1[0][1]**0.5)+epsilon)   
                        
                        

                        for k in range(1,self.input_nodes+1):
                            delta = self.learning_rate * self.val_in[k] * hiddenErrorGradient
                            self.momentum1[k][j]=eta*self.momentum1[k][j]+((1-eta)*(self.val_in[k] * hiddenErrorGradient)**2)
                            self.weights1[k][j] -=delta/((self.momentum1[k][1]**0.5)+epsilon)   
                #        delta = self.learning_rate * -1 * hiddenErrorGradient
                #print delta
                 #       self.thresholds[j] += delta
            for j in range(1, self.hidden_nodes+1):
                self.momentum2[j][1]=eta*self.momentum2[j][1]+((1-eta)*(self.values[j] * outputErrorGradient)**2)
                self.weights2[j][1] -= delta1[j]/((self.momentum2[j][1]**0.5)+epsilon) 
            # update the thresholds for the output node(s)
            #delta = self.learning_rate * -1 * outputErrorGradient
            #self.thresholds[3] += delta

                       

                # update the thresholds for the hidden nodes
            return math.pow(error,2)

    def calcError_adam(self,x,beta1,beta2):
            sumOfSquaredErrors = 0.0
            
            delta1 = nm.zeros(self.input_nodes+1)
        # we only look at the output nodes for error calculation
      
            error = x[2] - self.values[3]
            #print error


            outputErrorGradient = (self.values[3]) * error *(1-self.values[3])*-1
            #print outputErrorGradient
            delta = self.learning_rate * outputErrorGradient
           
            self.gradient2[0][1]=beta1*self.gradient2[0][1]+((1-beta1)* outputErrorGradient)
            self.momentum2[0][1]=beta2*self.momentum2[0][1]+((1-beta2)*(1 * outputErrorGradient)**2)
                        #print delta
            self.weights2[0][1] -=(self.learning_rate*self.gradient2[0][1])/((self.momentum2[0][1]**0.5)+epsilon)   
            
                        
            # now update the weights and thresholds
            for j in range(1, self.hidden_nodes+1):
            

                # first update for the hidden nodes to output nodes (1 layer)
                        delta1[j] = self.learning_rate * self.values[j] * outputErrorGradient
                        #print delta
                        
                        hiddenErrorGradient = (self.values[j])*(1-self.values[j]) * outputErrorGradient * self.weights2[j][1]
                        
                # and then update for the input nodes to hidden nodes
                        delta = self.learning_rate * hiddenErrorGradient
                        self.gradient1[0][j]=beta1*self.gradient1[0][j]+((1-beta1)* hiddenErrorGradient)
                        self.momentum1[0][j]=beta2*self.momentum1[0][j]+((1-beta2)*(1 * hiddenErrorGradient)**2)
                        #print delta
                        self.weights1[0][j] -=(self.learning_rate*self.gradient1[0][j])/((self.momentum1[0][j]**0.5)+epsilon)  
 #                       delta/((self.momentum1[0][1]**0.5)+epsilon)   
                        
                        

                        for k in range(1,self.input_nodes+1):
                            delta = self.learning_rate * self.val_in[k] * hiddenErrorGradient
                            self.gradient1[k][j]=beta1*self.gradient1[k][j]+((1-beta1)* hiddenErrorGradient*self.val_in[k])
                            self.momentum1[k][j]=beta2*self.momentum1[k][j]+((1-beta2)*(self.val_in[k] * hiddenErrorGradient)**2)
                            self.weights1[k][j] -=(self.learning_rate*self.gradient1[k][j])/((self.momentum1[k][j]**0.5)+epsilon)   
                #        delta = self.learning_rate * -1 * hiddenErrorGradient
                #print delta
                 #       self.thresholds[j] += delta
            for j in range(1, self.hidden_nodes+1):
                self.gradient2[j][1]=beta1*self.gradient2[j][1]+((1-beta1)* outputErrorGradient*self.values[j])
                self.momentum2[j][1]=beta2*self.momentum2[j][1]+((1-beta2)*(self.values[j] * outputErrorGradient)**2)
                        #print delta
                self.weights2[j][1] -=(self.learning_rate*self.gradient2[j][1])/((self.momentum2[j][1]**0.5)+epsilon)   
            # update the thresholds for the output node(s)
            #delta = self.learning_rate * -1 * outputErrorGradient
            #self.thresholds[3] += delta

                       

                # update the thresholds for the hidden nodes
            return math.pow(error,2)

net = neural_network(INPUT_NODES, HIDDEN_NODES, OUTPUT_NODES, LEARNING_RATE)
XOR_data=nm.genfromtxt('train.txt',delimiter=',')

#XOR_data = nm.array([[0,0,0], [0,1,1], [1,0,1], [1,1,0]])
cost_nag = []
print "SGD \n\n given :"
print net.weights1
print net.weights2


for i in range(MAX_ITERATIONS):
    Errors=0
    for x in XOR_data :
       net.process(1,x)
       err = net.calcError_sgd(x)
       Errors += err
       

     #  print net.weights1
      # print net.weights2
    #sumOfSquaredErrors/=2   
   # print Errors/8
    cost_nag.append(Errors*0.5/VALUE)    
print "SGD \n\n Corrected weight :"    
print net.weights1
print net.weights2
    
plt.plot(range(MAX_ITERATIONS), cost_nag, color='green', linewidth=3, label = "sgd")
plt.xlabel("iteration")
plt.ylabel("cost")

print "\n\n"
net = neural_network(INPUT_NODES, HIDDEN_NODES, OUTPUT_NODES, LEARNING_RATE)
XOR_data=nm.genfromtxt('train.txt',delimiter=',')
cost_nag = []
print "SGD with Momentum\n\nGiven :"
print net.weights1
print net.weights2


for i in range(MAX_ITERATIONS):
    Errors=0
    for x in XOR_data :
       net.process(1,x)
       err = net.calcError_momentum(x,0.3)
       Errors += err
       

     #  print net.weights1
      # print net.weights2
    #sumOfSquaredErrors/=2   
   # print Errors/8
    cost_nag.append(Errors*0.5/VALUE)    
print "SGD with Momentum\n\nCorrected weight :"
print net.weights1
print net.weights2
    
    
plt.plot(range(MAX_ITERATIONS), cost_nag, color='yellow', linewidth=3, label = "momentum")
plt.xlabel("iteration")
plt.ylabel("cost")
print "\n\n"
net = neural_network(INPUT_NODES, HIDDEN_NODES, OUTPUT_NODES, LEARNING_RATE)
XOR_data=nm.genfromtxt('train.txt',delimiter=',')
cost_nag = []
print "NAG \n\n Given"
print net.weights1
print net.weights2


for i in range(MAX_ITERATIONS):
    Errors=0
    for x in XOR_data :
       net.process(1,x)
       err = net.calcError_nag(x,0.3)
       Errors += err
       

     #  print net.weights1
      # print net.weights2
    #sumOfSquaredErrors/=2   
   # print Errors/8
    cost_nag.append(Errors*0.5/VALUE)    
    
print "NAG \n\n Corrected weight"
print net.weights1
print net.weights2
    
plt.plot(range(MAX_ITERATIONS), cost_nag, color='red', linewidth=3, label = "nag")
plt.xlabel("iteration")
plt.ylabel("cost")

print "\n\n"
net = neural_network(INPUT_NODES, HIDDEN_NODES, OUTPUT_NODES, LEARNING_RATE)
XOR_data=nm.genfromtxt('train.txt',delimiter=',')
cost_nag = []
print "Adagrad\n\nGiven"
print net.weights1
print net.weights2


for i in range(MAX_ITERATIONS):
    Errors=0
    for x in XOR_data :
       net.process(1,x)
       err = net.calcError_adagrad(x)
       Errors += err
       

     #  print net.weights1
      # print net.weights2
    #sumOfSquaredErrors/=2   
   # print Errors/8
    cost_nag.append(Errors*0.5/VALUE)    
    
print "Adagrad\n\nCorrected weight"
print net.weights1
print net.weights2
    
plt.plot(range(MAX_ITERATIONS), cost_nag, color='black', linewidth=3, label = "adagrad")
plt.xlabel("iteration")
plt.ylabel("cost")


print "\n\n"

net = neural_network(INPUT_NODES, HIDDEN_NODES, OUTPUT_NODES, LEARNING_RATE)
XOR_data=nm.genfromtxt('train.txt',delimiter=',')
cost_nag = []
print "RMSProp\n\nGiven"

print net.weights1
print net.weights2


for i in range(MAX_ITERATIONS):
    Errors=0
    
    for x in XOR_data :
       net.process(1,x)
       err = net.calcError_rmsprop(x,0.9)
       Errors += err
       

     #  print net.weights1
      # print net.weights2
    #sumOfSquaredErrors/=2   
   # print Errors/8
    cost_nag.append(Errors*0.5/VALUE)    
    
print "RMSProp\n\nCorrected weight"

print net.weights1
print net.weights2
    
plt.plot(range(MAX_ITERATIONS), cost_nag, color='orange', linewidth=3, label = "Rmsprop")
plt.xlabel("iteration")
plt.ylabel("cost")

print "\n\n"
net = neural_network(INPUT_NODES, HIDDEN_NODES, OUTPUT_NODES, LEARNING_RATE)

XOR_data=nm.genfromtxt('train.txt',delimiter=',')
cost_nag = []
print "Adam\n\nGiven"
print net.weights1
print net.weights2


for i in range(MAX_ITERATIONS):
    Errors=0
    
    for x in XOR_data :
       net.process(1,x)
       err=0 
       err = net.calcError_adam(x,0.9,0.99)
       Errors += err
       

     #  print net.weights1
      # print net.weights2
    #sumOfSquaredErrors/=2   
   # print Errors/8
    cost_nag.append(Errors*0.5/VALUE)    
    

print "Adam\n\nCorrected weight"
print net.weights1
print net.weights2
    
plt.plot(range(MAX_ITERATIONS), cost_nag, color='brown', linewidth=3, label = "Adam")
plt.xlabel("iteration")
plt.ylabel("cost")



print "\n\n"
net = neural_network(INPUT_NODES, HIDDEN_NODES, OUTPUT_NODES, LEARNING_RATE)
XOR_data=nm.genfromtxt('train.txt',delimiter=',')
cost_nag = []
print "Adadelta\n\nGiven"
print net.weights1
print net.weights2

for i in range(MAX_ITERATIONS):
    Errors=0
    if i%W==0 :
        net.init_arr();
       
    for x in XOR_data :
       net.process(1,x)
       err=0 
       err = net.calcError_adagrad(x)
       Errors += err
       

     #  print net.weights1
      # print net.weights2
    #sumOfSquaredErrors/=2   
   # print Errors/8
    cost_nag.append(Errors*0.5/VALUE)    
    

print "Adadelta\n\nCorrected weight"
print net.weights1
print net.weights2

plt.plot(range(MAX_ITERATIONS), cost_nag, color='violet', linewidth=3, label = "Adadelta")
plt.xlabel("iteration")
plt.ylabel("cost")



plt.legend(['SGD','momentum','NAG','Adagrad','Rmsprop','Adam','Adadelta'], loc='upper left')
plt.show()

boundary1 = nm.linspace(-.2,1.2,100)
plt.plot(boundary1,boundary1+0.4,c='black')
plt.plot(boundary1,boundary1-0.6,c='black')
plt.scatter(1,1,c='red',s=100)
plt.scatter(0,0,c='red',s=100)
plt.scatter(0,1,s=100)
plt.scatter(1,0,s=100)
plt.fill_between(x=boundary1,y1=boundary1+.4,y2=boundary1+3,alpha=.2,color='blue')
plt.fill_between(x=boundary1,y1=boundary1-.6,y2=boundary1-2,alpha=.2,color='blue')
plt.fill_between(x=boundary1,y1=boundary1+.4,y2=boundary1-0.6,alpha=.2,color='red')
plt.xlim(-.2,1.2)
plt.ylim(-.1,1.1)
plt.show()

