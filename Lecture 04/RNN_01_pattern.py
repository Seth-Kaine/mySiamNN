# Lecture 04 UVM
# Josh Bongard
# 2016
# Evolutionary Robotics
# NN
# Backpropagation
# learning network
# C:\Users\Seth>cd "Desktop\Archives\entertain\tutorial NEAT\Best\Josh_Bongard\cours\results\Lecture 04\

import random
import math
truthTableOR = [
[0,0],
[0,1],
[1,0],
[1,1],
]

tableTarget = [
[0,0],
[1,1],
[1,1],
[1,1],

]

# Semi supervised learning
# - Reward returned for a series of input patterns.
"""
Motors react to sensors
 - take the sensors and process the result next applied to motors
 - do a series of sensors getResults applied to the current motors
 - evaluate the total distance for this series of mesures
 Next :
 - evolve and compare pattern of sensors by steps and by group
 - fitest pattern evolve!

 RNN throught NEAT evolution network
 Genotype : storage of genetic information (data structure that contains the numbers that we are going to be evolving)
 Phenotype : system produced by that information (the thing that we are going to mesure like a NN, a robot, etc.)


"""

population_size = 1

# Process of creation for the fitest
"""
1. Population initialization
2. Fitness Scoring
3. Selection (of the fitest)
4. Variation (reproduction of the fitest parents and go to 2. loop for a new generation)
5. Terminaison
"""


# Evolution
"""
the Genetic Algorithm
recombination
population_size >1
notion of sex with parents produce children with mutation and split their own genes in two to recombine the new individu : their child
- each individu has its own  Fitness Scoring
# Generation
table of the fitness for each genes value
- fitness landscape
- recombination of the two parents lands between the two parents gene values position in the graphic landscape (recombination hotspots -> read essais)
"""
"""
The Evolution Strategy
none randomness applies it to mutation not to recombination

step_size or magnitude event that the evolution should have ex. 0.1
mutation with the stepSize +/-

the harder is it to climb the hill the smaller must be the change you make - stepSize must be smaller

step sizes or stategy parameters
all numbers have their own magnitude mutation stategy

# Genetic programing - symbolic regression
evolving program or code tree
genotype (blueprint) is represented as a tree rather than a string of numbers
parents recombine branches to create children
Possible branch nodes : sin(a), cos(a), plus(a,b), minus(a,b), mult(a,b), div(a,b), pow(a,b)
possible terminal nodes : x,y, ... 0.1, -3.0, ...

higher the fitness of a program -> on lower levels are the modifications  of mutations or recombination happening  onto the end branches

Sum all the numbers from the pattern
Error =  Sum as for loop for all the points (x;y) ->(sqrt(pow((xi-xi'),2)+pow(yi-yi',2)))/n for total numbers of entries of all the x & y points, this is nedded to show error and choose the high fitness algorithm/equation -> exclude points that don't fit the equation

noise could be added to fit more the final equation
lecture 05 33:24
"""




class Node:
    """
    A Node
    # node class has a value, a type [input or output]
    # connection weight class has a value and is bind two nodes
    # + threshold 0 < x < 1
    """

    def __init__(self, id, value, type ):
        # Redirect output to a queue
        self.id = id
        self.value = value
        self.type = type

    def __call__(self, value):
        return "hello"
        #additional logic
    def setValue(self, value):
        self.value = value
    def getValue(self):
        return self.value

    def getType(self):
        return self.type

class Connection:
    """
    # connection weight class has a value and is bind two nodes
    # + threshold 0 < x < 1
    """

    def __init__(self, id, weight, nodeIN, nodeOUT ):
        # Redirect output to a queue
        self.id = id
        self.weight = weight
        self.nodeIN = nodeIN
        self.nodeOUT = nodeOUT

    def setWeight(self, value):
        self.weight = value
    def getWeight(self):
        return self.weight

    def __getitem__(self, key):
        #if key in self:
            # the super() call works here because we are
            # subclassing dict, which supports __getitem__
            #return super().__getitem__(key)
        if self.nodeIN is not None:
            val = self.nodeIN
            #self[key] = val
            return val
        raise KeyError(key)
    def getNodes(self):
        return (self.nodeIN,self.nodeOUT)

class Trainer:
    """
    the Trainer :
    - set the data set
    - evaluate the data set and the
    """
    def __init__(self, net, inputsList, resultsList, ntimes):
        self.RMSerror = 0
        self.epoch

"""
 * bkp_evaluate - Evaluate but don't learn the current input set.
 * This is usually preceded by a call to bkp_set_input() and is
 * typically called after the training set (epoch) has been learned.
 *
 * If you give eoutputvals as NULL then you can do a bkp_query() to
 * get the results.
 *
 * If you give the address of a buffer to return the results of the
 * evaluation (eoutputvals != NULL) then the results will copied to the
 * eoutputvals buffer.
 *
 * Return Values:
 * int 0: Success
 *    -1: Error, errno is:
 *        ENOENT if no bkp_create_network() has been done yet.
 *        ESRCH if no bkp_set_input() has been done yet.
 *        ENODEV if both bkp_create_network() and bkp_set_input()
 *               have been done but bkp_earn() has not been done
 *               yet (ie; neural net has not had any training).
 *        EINVAL if sizeofoutputvals is not the same as the
 *               size understood according to n. This is to help
 *               prevent buffer overflow during copying.
"""


"""
 * bkp_forward - This makes a pass from the input units to the hidden
 * units to the output units, updating the hidden units, output units and
 * other components. This is how the neural network is run in order to
 * evaluate a set of input values to get output values.
 * When training the neural network, this is the first step in the
 * backpropagation algorithm.
 """

"""
 * bkp_backward - This is the 2nd half of the backpropagation algorithm
 * which is carried out immediately after bkp_forward() has done its
 * step of calculating the outputs. This does the reverse, comparing
 * those output values to those given as targets in the training set
 * and updating the weights and other components appropriately, which
 * is essentially the training of the neural network.
"""

class Network:
    """
    A Node
    # node class has a value, a type [input or output]
    # connection weight class has a value and is bind two nodes
    # + threshold 0 < x < 1
    """
    # conf list [ins, outs, hidden, stepSize, momemtum, cost]
    configDefault = [2,2,0,.2,.0,.0]
    ntimes = 1
    def __init__(self, config=configDefault, inputsList=[] ):
        # for num in config:
            # print num
            # for i in range(int(num)):
                # print "node"
        self.numInputs = config[0]
        self.out = []
        self.bind = []
        self.numOutputs = config[1]
        self.stepVal = config[3]
        self.inputs = inputsList
        self.results = []
        self.init_net()

    def init_net(self):
        print "create_net"
        # create inputs
        if self.inputs:
            #print self.inputsList
            self.numInputs = len(self.inputs)
        else:
            self.inputs.extend([1 for val in range(self.numInputs)])
        # create Node
        #for node in self.inputs:
        for i, nod in enumerate(self.inputs):
            print "%s - %s " % (i,nod)
            noded = int(nod) #type(nod, "Integer") #
            node = Node(noded,self.inputs[noded],"input")
            #self.inp.append(node)
            #print node.id
            self.inputs[noded] = node
        for node in range(self.numOutputs):
            node = Node(node,0,"output")
            #print node.id
            self.out.append(node)
        for i in range(self.numInputs):
            for j in range(self.numOutputs):
                #print "Connection %d %d " % (i,j)
                weight = round(random.random(),1)
                id = "%d-%d" % (i,j)
                connect = Connection(id,weight,self.inputs[i],self.out[j])

                self.bind.append(connect)
    def set_training_set(self, inputsList, resultsList):
        return 0
    def learn(self, ntimes=ntimes):
        return 0
    def set_inputs(self, value):
        return "hello"
    def process_out(self):

        ## for c in self.bind: # sum of all out
        for b in self.bind:

                print "Weight : %s" % b.getWeight()
                print "nodeIN value : %s" % b.nodeIN.getValue()
                print "nodeOUT.id : %s" % b.nodeOUT.id
                #node03.setValue(a*connect13.getWeight() + b*connect23.getWeight())


        return "hello"
    def evaluate(self, value):
        return "hello"

    def export_weights(self, value):
        return "array 2D"

def main():
    # import DATA

    #Create model
    #net = Network()
    net = Network(inputsList=[1.0,1.0])
    print [node.id for node in net.inputs]
    print [c.weight for c in net.bind]
    net.process_out()

    # TRAIN

    # test trained




def sigmoid(x):
    return 1/(1+math.exp(-x))



def training(net):
    return 0

def backprop(in1, in2, no_of_out):
    node01 = Node(in1,"input")
    node02 = Node(in2,"input")
    node03 = Node(0,"output")
    node04 = Node(0,"output")
    #print node01.getValue()
    weight1 = round(random.random(),1)
    weight2 = round(random.random(),1)
    weight3 = round(random.random(),1)
    weight4 = round(random.random(),1)
    connect13 = Connection(weight1, node01,node03)
    connect14 = Connection(weight2, node01,node04)
    connect23 = Connection(weight3, node02,node03)
    connect24 = Connection(weight4, node02,node04)
    a = connect13[0].getValue()
    b = connect23[0].getValue()
    c = connect14[0].getValue()
    d = connect24[0].getValue()
    #print connect13.getWeight()
    #print a
    # print b.getValue()


    thresholdX1 = 1 # between .5 and .8 "AND table" under < .5 "OR table"
    thresholdX2 = 1
    node03.setValue(a*connect13.getWeight() + b*connect23.getWeight())
    node04.setValue(c*connect14.getWeight() + d*connect24.getWeight())

    x1 = node03.getValue()
    x2 = node04.getValue()

    if x1 > thresholdX1:
        x1 = 1
    elif x1 <= thresholdX1:
        x1 = 0
    if x2 > thresholdX2:
        x2 = 1
    elif x2 <= thresholdX2:
        x2 = 0

    print (x1,x2)

# for row in truthTableOR:
    # backprop(row[0],row[1])

if __name__ == '__main__':
    ## last config before run the app : create a class with prog to run
    #tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
    print "fff"
    main()
