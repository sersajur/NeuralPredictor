from IPredictor import IPredictor
from pybrain.structure import FeedForwardNetwork
from pybrain.structure.modules import(
    BiasUnit,
    TanhLayer,
    SoftmaxLayer,
    LinearLayer
)
from pybrain.structure.connections import FullConnection
from pybrain.datasets import SupervisedDataSet, ClassificationDataSet
from pybrain.tests.helpers import gradientCheck
from pybrain.supervised import RPropMinusTrainer
from pybrain.tools.validation import ModuleValidator
import os
import pickle
from DayFeatureExpert import DayFeatureExpert


class FNNPredictor(IPredictor):

    @property
    def name(self):
        return "FNN"

    def __init__(self, gl_normaliser=lambda x: float(x) / 300, time_normaliser=lambda t: float(t) / (24*3600)):
        self.m_net = FNNPredictor._CreateFeedforwardNN()
        self.m_gl_normaliser = gl_normaliser
        self.m_time_normaliser = time_normaliser

    def Load(self, dump_file):
        if not os.path.isfile(dump_file):
            return False

        pkl_file = open(dump_file, 'rb')
        net = pickle.load(pkl_file)
        if not FNNPredictor._IsLoadedNetValid(net):
            return False

        self.m_net = net
        return True

    def Predict(self, pt_record):
        input = FNNPredictor._ComposeInputFromPtRecord(pt_record, self.m_time_normaliser, self.m_gl_normaliser)
        output = self.m_net.activate(input)
        print output
        return output[0] > output[1]

    def Train(self, dataset, error_observer, logger, dump_file):
        gradientCheck(self.m_net)

        net_dataset = SupervisedDataSet(self.m_net.indim, self.m_net.outdim)
        for record in dataset:
            input = FNNPredictor._ComposeInputFromPtRecord(record, self.m_time_normaliser, self.m_gl_normaliser)
            if DayFeatureExpert.IsHypoglycemia(record):
                target_out = [1, 0]
            else:
                target_out = [0, 1]
            net_dataset.addSample(input, target_out)

        train_dataset, test_dataset = net_dataset.splitWithProportion(0.8)
        train_h_count = sum(1 for x in train_dataset['target'] if x[0] == 1)
        test_h_count = sum(1 for x in test_dataset['target'] if x[0] == 1)
        logger("Train dataset statistics: " + str(train_h_count) + ' / ' + str(len(train_dataset) - train_h_count))
        logger("Test dataset statistics: " + str(test_h_count) + ' / ' + str(len(test_dataset) - test_h_count))
        trainer = RPropMinusTrainer(self.m_net, dataset=train_dataset, momentum=0.8, learningrate=0.01, lrdecay=0.95, weightdecay=0.1, verbose=True)
        validator = ModuleValidator()

        train_error = []
        test_error = []
        for i in range(0, 100):
            trainer.trainEpochs(1)
            train_error.append(validator.MSE(self.m_net, train_dataset))
            test_error.append(validator.MSE(self.m_net, test_dataset))
            error_observer(train_error, test_error)
            gradientCheck(self.m_net)

        logger("Train error: " + str(train_error[-1]))
        logger("Test error: " + str(test_error[-1]))
        dump_file = open(dump_file, 'wb')
        pickle.dump(self.m_net, dump_file)

    def ForgetKnowledge(self):
        self.m_net = FNNPredictor._CreateFeedforwardNN()

    @staticmethod
    def _CreateFeedforwardNN():
        net = FeedForwardNetwork()

        net.addInputModule(LinearLayer(12, name='inLayer'))
        net.addModule(TanhLayer(3, name='hiddenLayer'))
        net.addOutputModule(SoftmaxLayer(2, name='outLayer'))
        net.addModule(BiasUnit(name='hiddenBiasUnit'))
        net.addModule(BiasUnit(name='outBiasUnit'))

        net.addConnection(FullConnection(net['hiddenBiasUnit'], net['hiddenLayer'], name='b-hidden-Connection'))
        net.addConnection(FullConnection(net['inLayer'], net['hiddenLayer'], name='in-hidden-Connection'))
        net.addConnection(FullConnection(net['outBiasUnit'], net['outLayer'], name='b-out-Connection'))
        net.addConnection(FullConnection(net['hiddenLayer'], net['outLayer'], name='hidden-out-Connection'))
        net.sortModules()

        return net

    @staticmethod
    def _IsLoadedNetValid(net):
        if net.indim != 12 or net.outdim != 2:
            return False
        return True

    @staticmethod
    def _ComposeInputFromPtRecord(pt_record, time_normaliser, gl_normaliser):
        gl_rises = pt_record.GetGlRises()
        assert len(gl_rises) == 3

        input = list()
        for gl_rise in gl_rises:
            input.append(time_normaliser(gl_rise[0][0].total_seconds()))
            input.append(gl_normaliser(gl_rise[0][1]))
            input.append(time_normaliser(gl_rise[1][0].total_seconds()))
            input.append(gl_normaliser(gl_rise[1][1]))

        return input
