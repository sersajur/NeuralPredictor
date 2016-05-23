#!/usr/bin/env python
# -*- coding: utf-8 -*-

import RecurentNN
from DataReader import ReadDataSet
from pybrain.tests.helpers import gradientCheck
from pybrain.datasets import SequentialDataSet
from pybrain.supervised import RPropMinusTrainer, BackpropTrainer
from pybrain.tools.validation import ModuleValidator
from matplotlib import pyplot as plt
from DayFeatureEx import DayFeatureEx

net = RecurentNN.CreateRecurentNN()
gradientCheck(net)

raw_data_set = ReadDataSet(u"D:\GDrive\Диплом 2\DataPreparation\output1\\not_fixed_dataset.xlsx")
net_dataset = SequentialDataSet(4, 1)
for record in raw_data_set:
    net_dataset.newSequence()

    gl_raises = record.GetGlRises()
    gl_min = record.GetNocturnalMinimum()

    for gl_raise in gl_raises:
        net_dataset.addSample([gl_raise[0][0].total_seconds() / 3600, gl_raise[0][1], gl_raise[1][0].total_seconds() / 3600, gl_raise[1][1]], gl_min[1])

train_dataset, test_dataset = net_dataset.splitWithProportion(0.8)

trainer = BackpropTrainer(net, dataset=train_dataset, momentum=0, learningrate=0.005)
validator = ModuleValidator()

train_error = []
test_error = []
plt.ion()
plt.show()
for i in range(0, 100):
    trainer.trainEpochs(5)
    train_error.append(validator.MSE(net, train_dataset)) # here is validate func, think it may be parametrised by custom core function
    test_error.append(validator.MSE(net, test_dataset))
    print train_error
    print test_error
    plt.cla()
    h_train_er = plt.plot(train_error, 'r')
    h_test_er = plt.plot(test_error, 'g')
    plt.draw()
    plt.pause(0.1)
    gradientCheck(net)
