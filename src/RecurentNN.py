
from pylab import plot, hold, show
from scipy import sin, rand, arange
from pybrain.datasets            import SequenceClassificationDataSet
from pybrain.structure           import RecurrentNetwork, LinearLayer, TanhLayer, FullConnection
from pybrain.structure.modules   import LSTMLayer, SoftmaxLayer
from pybrain.supervised          import RPropMinusTrainer
from pybrain.tools.validation    import testOnSequenceData
from pybrain.tools.shortcuts     import buildNetwork

from DayFeatureEx import DayFeatureEx

def CreateRecurentNN():
    net = RecurrentNetwork()
    net.addInputModule(LinearLayer(4, name='in'))
    net.addModule(TanhLayer(2, name='hidden'))
    net.addOutputModule(LinearLayer(1, name='out'))
    net.addConnection(FullConnection(net['in'], net['hidden'], name='fc1'))
    net.addConnection(FullConnection(net['hidden'], net['out'], name='fc2'))
    net.addRecurrentConnection(FullConnection(net['hidden'], net['hidden'], name='rc3'))
    net.sortModules()
    return net

def TrainRecurentNN(net, data_set):
    return
