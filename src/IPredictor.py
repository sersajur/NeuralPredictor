from abc import ABCMeta, abstractmethod


class IPredictor():
    __metaclass__ = ABCMeta

    @property
    @abstractmethod
    def name(self):
        """Predictors name"""

    @abstractmethod
    def Predict(self, pt_record):
        """Predict hypoglycemia"""

    @abstractmethod
    def Train(self, train_dataset, error_observer,  dump_file):
        """Train on train_dataset"""

    @abstractmethod
    def Load(self, dump_file):
        """Load trained predictor from dump_file"""

    @abstractmethod
    def ForgetKnowledge(self):
        """Reset trained memory"""
