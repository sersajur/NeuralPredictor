from IPredictor import IPredictor


class RNNPredictorEx(IPredictor):

    @property
    def name(self):
        return "RNNPredictor + demographic data"

    def Load(self, dump_file):
        return False

    def Predict(self, pt_record):
        return True

    def Train(self, train_dataset, error_observer, logger, dump_file):
        return

    def ForgetKnowledge(self):
        return
