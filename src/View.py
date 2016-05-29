#!/usr/bin/env python
# -*- coding: utf-8 -*-
from socket import meth

import wx
from PlotWidget import PlotWidget
from datetime import datetime

class View(wx.Frame):
    def __init__(self):
        self.app = wx.App(False)

        wx.Frame.__init__(self, None, title="Nocturnal hypoglycemia prediction", size=(1100, 800))
        panel = wx.Panel(self)

        label = wx.StaticText(panel, label="Train dataset path:", pos=(10, 10))
        self.m_train_dataset_path_edit = wx.TextCtrl(panel, pos=(10, 30), size=(800, 20))
        self.m_train_dataset_path_edit.SetValue(u"D:\GDrive\Диплом 2\DataPreparation\output1\\fixed_dataset+_validate_known.xlsx")

        label = wx.StaticText(panel, label="Test dataset path:", pos=(10, 60))
        self.m_test_dataset_path_edit = wx.TextCtrl(panel, pos=(10, 80), size=(800, 20))
        self.m_test_dataset_path_edit.SetValue(u"D:\GDrive\Диплом 2\DataPreparation\output1\\fixed_dataset+_train.xlsx")

        label = wx.StaticText(panel, label="Choose method:", pos=(10, 110))
        self.m_predictor_combobox = wx.ComboBox(panel, pos=(10, 130), size=wx.DefaultSize, choices=['stub RNN', 'stub RNN+demographic'], style=wx.CB_READONLY)

        self.m_test_button = wx.Button(panel, label="Test", pos=(10, 160), size=wx.DefaultSize)
        self.m_train_button = wx.Button(panel, label="Train", pos=(110, 160), size=wx.DefaultSize)

        label = wx.StaticText(panel, label="Prediction log:", pos=(10, 200))
        self.m_log = wx.TextCtrl(panel, pos=(10, 220), size=(360, 480), style=wx.TE_MULTILINE | wx.CB_READONLY)

        self.plot = PlotWidget(panel, pos=(400, 220), size=(10, 20))
    # Custom bindings
    def SetOnTestAction(self, on_test_action):
        self.Bind(wx.EVT_BUTTON, lambda x: on_test_action(), self.m_test_button)

    def SetOnTrainAction(self, on_train_action):
        self.Bind(wx.EVT_BUTTON, lambda x: on_train_action(), self.m_train_button)

    # Custom functions
    def UpdateMethodList(self, i_method_list):
        self.m_predictor_combobox.Clear()
        for method in i_method_list:
            self.m_predictor_combobox.Append(method)
        self.m_predictor_combobox.SetSize((200, -1))

    def GetSelectedMethodName(self):
        return self.m_predictor_combobox.GetValue()

    def GetTrainDataPath(self):
        return self.m_train_dataset_path_edit.GetValue()

    def GetTestDataPath(self):
        return self.m_test_dataset_path_edit.GetValue()

    def PrintToLog(self, i_text):
        self.m_log.AppendText('[' + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '] ' + i_text + '\n')

    def PlotGraph(self, x_arr, y_arr, col):
        self.plot.draw(x_arr, y_arr, col)

    def ClearGraph(self):
        self.plot.clear()

    # Run (all settings must be done before this function run)
    def Run(self):
        self.Show(True)
        self.app.MainLoop()
