import numpy as np
from visualizations import plot_pr_curve

class PrecisionRecallCurve(object):
    def __init__(self):
        super(PrecisionRecallCurve, self).__init__()
        self.reset_states()

    def reset_states(self):
        self.y_true = []
        self.y_score = []
        self.n_gt_boxes = 0

    def update_states(self, y_true, y_score, n_gt_boxes):
        self.y_true.extend(y_true)
        self.y_score.extend(y_score)
        self.n_gt_boxes += n_gt_boxes

    def result(self):

        def average_precision(precisions, recalls):
            mrec = []
            mrec.append(0)
            [mrec.append(e) for e in recalls]
            mrec.append(1)
            mpre = []
            mpre.append(0)
            [mpre.append(e) for e in precisions]
            mpre.append(0)
            for i in range(len(mpre) - 1, 0, -1):
                mpre[i - 1] = max(mpre[i - 1], mpre[i])
            poi = []
            for i in range(len(mrec) - 1):
                if mrec[1:][i] != mrec[0:-1][i]:
                    poi.append(i + 1)
            ap = 0
            for i in poi:
                ap = ap + np.sum((mrec[i] - mrec[i - 1]) * mpre[i])
            return [ap, mpre[0:len(mpre) - 1], mrec[0:len(mpre) - 1], poi]

        idx = np.argsort(self.y_score)
        sorted_y_true = np.array(self.y_true)[idx[::-1]]
        cum_tp = np.cumsum(sorted_y_true)
        cum_fp = np.cumsum(1 - sorted_y_true)
        precisions = cum_tp / (cum_tp + cum_fp)
        recalls = cum_tp / self.n_gt_boxes

        ap, y, x, _ = average_precision(precisions, recalls)
        plot_pr_curve(y, x, ap)
        
        return ap