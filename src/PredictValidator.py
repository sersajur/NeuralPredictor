from math import sqrt


class PredictValidator:

    @staticmethod
    def _TP(expected_arr, actual_arr):
        return sum(1 for xy in map(lambda x, y: (x, y), expected_arr, actual_arr) if xy[0]==xy[1]==True)

    @staticmethod
    def _TN(expected_arr, actual_arr):
        return sum(1 for xy in map(lambda x, y: (x, y), expected_arr, actual_arr) if xy[0]==xy[1]==False)

    @staticmethod
    def _FP(expected_arr, actual_arr):
        return sum(1 for xy in map(lambda x, y: (x, y), expected_arr, actual_arr) if xy[0]!=xy[1] and xy[1]==True)

    @staticmethod
    def _FN(expected_arr, actual_arr):
        return sum(1 for xy in map(lambda x, y: (x, y), expected_arr, actual_arr) if xy[0]!=xy[1] and xy[1]==False)

    def __init__(self, expected_arr, actual_arr):
        self.m_TP = PredictValidator._TP(expected_arr, actual_arr)
        self.m_TN = PredictValidator._TN(expected_arr, actual_arr)
        self.m_FP = PredictValidator._FP(expected_arr, actual_arr)
        self.m_FN = PredictValidator._FN(expected_arr, actual_arr)

    def TruePositiveRate(self):
        return float(self.m_TP) / (self.m_TP + self.m_FN)

    def TrueNegativeRate(self):
        return float(self.m_TN) / (self.m_TN + self.m_FP)

    def PositivePredictiveValue(self):
        return float(self.m_TP) / (self.m_TP + self.m_FP)

    def NegativePredictiveValue(self):
        return float(self.m_TN) / (self.m_TN + self.m_FN)

    def F1Score(self):
        return 2 * float(self.m_TP) / (2 * self.m_TP + self.m_FP + self.m_FN)

    def F1ScoreDual(self):
        return 2 * float(self.m_TN) / (2 * self.m_TN + self.m_FN + self.m_FP)

    def MatthewsCorCoef(self):
        tp = float(self.m_TP)
        tn = float(self.m_TN)
        fp = float(self.m_FP)
        fn = float(self.m_FN)
        return (tp * tn - fp * fn) / sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))