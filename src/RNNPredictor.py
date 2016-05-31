#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pybrain.structure import RecurrentNetwork, LinearLayer, TanhLayer, FullConnection, BiasUnit, SoftmaxLayer, FeedForwardNetwork, SigmoidLayer
from pybrain.tests.helpers import gradientCheck
from pybrain.datasets import SequentialDataSet, SequenceClassificationDataSet
from pybrain.supervised import RPropMinusTrainer, BackpropTrainer
from pybrain.tools.validation import ModuleValidator
from matplotlib import pyplot as plt
from DayFeatureEx import DayFeatureEx
from DayFeatureExpert import DayFeatureExpert

from IPredictor import IPredictor
import pickle
import os


class RNNPredictor(IPredictor):

    def __init__(self):
        self.m_net = RNNPredictor._CreateRecurentNN()

    @property
    def name(self):
        return "RNN"

    @staticmethod
    def _IsLoadedNetValid(net):
        if net.indim != 4 or net.outdim != 2:
            return False
        return True

    def Load(self, dump_file):
        if not os.path.isfile(dump_file):
            return False

        pkl_file = open(dump_file, 'rb')
        net = pickle.load(pkl_file)
        if not RNNPredictor._IsLoadedNetValid(net):
            return False

        self.m_net = net
        return True

    def Predict(self, pt_record):
        gl_rises = pt_record.GetGlRises()
        self.m_net.reset()
        output = 0
        for gl_rise in gl_rises:
            net_input = [
                gl_rise[0][0].total_seconds() / (24*3600),
                gl_rise[0][1] / 300,
                gl_rise[1][0].total_seconds() / (24*3600),
                gl_rise[1][1] / 300
            ]
            output = self.m_net.activate(net_input)
            print output
        return output[0] > output[1]

    def Train(self, dataset, error_observer, logger, dump_file):
        gradientCheck(self.m_net)

        net_dataset = SequenceClassificationDataSet(4, 2)
        for record in dataset:
            net_dataset.newSequence()

            gl_raises = record.GetGlRises()
            gl_min = record.GetNocturnalMinimum()

            if DayFeatureExpert.IsHypoglycemia(record):
                out_class = [1, 0]
            else:
                out_class = [0, 1]

            for gl_raise in gl_raises:
                net_dataset.addSample([gl_raise[0][0].total_seconds() / (24*3600), gl_raise[0][1] / 300, gl_raise[1][0].total_seconds() / (24*3600), gl_raise[1][1] / 300] , out_class)

        train_dataset, test_dataset = net_dataset.splitWithProportion(0.8)

        trainer = RPropMinusTrainer(self.m_net, dataset=train_dataset, momentum=0.8, learningrate=0.3, lrdecay=0.9, weightdecay=0.01, verbose=True)
        validator = ModuleValidator()

        train_error = []
        test_error = []
        for i in range(0, 80):
            trainer.trainEpochs(1)
            train_error.append(validator.MSE(self.m_net, train_dataset)) # here is validate func, think it may be parametrised by custom core function
            test_error.append(validator.MSE(self.m_net, test_dataset))
            print train_error
            print test_error
            error_observer(train_error, test_error)
            gradientCheck(self.m_net)

        dump_file = open(dump_file, 'wb')
        pickle.dump(self.m_net, dump_file)

    def ForgetKnowledge(self):
        self.m_net = RNNPredictor._CreateRecurentNN()

    @staticmethod
    def _CreateRecurentNN():
        net = RecurrentNetwork()
        net.addInputModule(LinearLayer(4, name='in'))
        net.addModule(BiasUnit(name='hidden_bias'))
        net.addModule(TanhLayer(13, name='hidden'))
        #net.addModule(BiasUnit(name='out_bias'))
        net.addOutputModule(SoftmaxLayer(2, name='out_class'))
        #net.addOutputModule(LinearLayer(1, name='out_predict'))
        #net.addConnection(FullConnection(net['out_bias'], net['out_predict']))
        net.addConnection(FullConnection(net['hidden_bias'], net['hidden']))
        net.addConnection(FullConnection(net['in'], net['hidden'], name='fc1'))
        net.addConnection(FullConnection(net['hidden'], net['out_class'], name='fc2'))
        #net.addConnection(FullConnection(net['hidden'], net['out_predict'], name='fc3'))
        net.addRecurrentConnection(FullConnection(net['hidden'], net['hidden'], name='rc3'))
        net.sortModules()
        return net


