#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __builtin__ import staticmethod

from View import View
from os import mkdir
from os.path import isdir, isfile
from RNNPredictor import RNNPredictor
from RNNPredictorEx import RNNPredictorEx
from DataReader import ReadDataSet
from DayFeatureExpert import DayFeatureExpert
from PredictValidator import PredictValidator
from numpy import arange
import thread

class PredictorsManager:
    g_view = View()
    g_predictors = dict()
    g_dumps_root = u"D:\GDrive\Диплом 2\HypohlycemiaPrediction\Predictors"

    @staticmethod
    def _ChooseAppropriateMethod(i_method_name):
        assert i_method_name in PredictorsManager.g_predictors.keys()

        return PredictorsManager.g_predictors[i_method_name]

    @staticmethod
    def _GetDumpDirPath(method_name):
        return PredictorsManager.g_dumps_root + "\\" + method_name

    @staticmethod
    def _GetDumpFile(method_name):
        return PredictorsManager._GetDumpDirPath(method_name) + "\\" + "dump.p"

    @staticmethod
    def _RegisterPredictor(i_new_predictor):
        assert i_new_predictor.name not in PredictorsManager.g_predictors.keys()

        PredictorsManager.g_predictors[i_new_predictor.name] = i_new_predictor
        dump_dir = PredictorsManager._GetDumpDirPath(i_new_predictor.name)
        if not isdir(dump_dir):
            mkdir(dump_dir)

    @staticmethod
    def _ValidatePredictor(predictor):
        test_data_path = PredictorsManager.g_view.GetTestDataPath()
        if not isfile(test_data_path):
            PredictorsManager.g_view.PrintToLog("Test dataset file does not exist!")
            return

        data_set = ReadDataSet(test_data_path)
        expected_arr = map(lambda x: DayFeatureExpert.IsHypoglycemia(x), data_set)
        actual_arr = map(lambda x: predictor.Predict(x), data_set)
        validator = PredictValidator(expected_arr, actual_arr)

        PredictorsManager.g_view.PrintToLog("TPR: " + str(validator.TruePositiveRate()))
        PredictorsManager.g_view.PrintToLog("TNR: " + str(validator.TrueNegativeRate()))
        PredictorsManager.g_view.PrintToLog("PPV: " + str(validator.PositivePredictiveValue()))
        PredictorsManager.g_view.PrintToLog("NPV: " + str(validator.NegativePredictiveValue()))
        PredictorsManager.g_view.PrintToLog("F1 score: " + str(validator.F1Score()))

    @staticmethod
    def _OnTrainAction():
        method_name = PredictorsManager.g_view.GetSelectedMethodName()
        predictor = PredictorsManager._ChooseAppropriateMethod(method_name)

        train_data_path = PredictorsManager.g_view.GetTrainDataPath()
        if not isfile(train_data_path):
            PredictorsManager.g_view.PrintToLog("Train dataset file does not exist!")
            return

        data_set = ReadDataSet(train_data_path)

        def error_reporter(train_err_list, test_err_list):
            PredictorsManager.g_view.ClearGraph()
            PredictorsManager.g_view.PlotGraph(arange(0, len(train_err_list)), train_err_list, 'r')
            PredictorsManager.g_view.PlotGraph(arange(0, len(test_err_list)), test_err_list, 'g')

        def training_task_function():
            PredictorsManager.g_view.PrintToLog("=============================")
            PredictorsManager.g_view.PrintToLog("Predictor: " + predictor.name)
            PredictorsManager.g_view.PrintToLog("Training starts")
            predictor.Train(data_set, error_reporter, PredictorsManager._GetDumpFile(method_name))
            PredictorsManager.g_view.PrintToLog("Testing predictor:")
            PredictorsManager._ValidatePredictor(predictor)
            PredictorsManager.g_view.PrintToLog("Training ends")

        thread.start_new_thread(training_task_function, ())

    @staticmethod
    def _OnTestAction():
        method_name = PredictorsManager.g_view.GetSelectedMethodName()
        predictor = PredictorsManager._ChooseAppropriateMethod(method_name)

        dump_file_name = PredictorsManager._GetDumpFile(method_name)
        if not isfile(dump_file_name):
            PredictorsManager.g_view.PrintToLog("Dump file does not exist: " + dump_file_name)
            return
        if not predictor.Load(dump_file_name):
            PredictorsManager.g_view.PrintToLog("Could not load trained predictor from file: " + dump_file_name)
            return

        def testing_task_function():
            PredictorsManager.g_view.PrintToLog("=============================")
            PredictorsManager.g_view.PrintToLog("Predictor: " + predictor.name)
            PredictorsManager.g_view.PrintToLog("Testing predictor:")
            PredictorsManager._ValidatePredictor(predictor)


        thread.start_new_thread(testing_task_function, ())

    @staticmethod
    def _Init():
        PredictorsManager._RegisterPredictor(RNNPredictor())
        PredictorsManager._RegisterPredictor(RNNPredictorEx())

        PredictorsManager.g_view.UpdateMethodList(PredictorsManager.g_predictors.keys())

        PredictorsManager.g_view.SetOnTrainAction(PredictorsManager._OnTrainAction)
        PredictorsManager.g_view.SetOnTestAction(PredictorsManager._OnTestAction)

    @staticmethod
    def Run():
        PredictorsManager._Init()
        PredictorsManager.g_view.Run()


PredictorsManager.Run()
