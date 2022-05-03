import numpy as np
import torch.nn as nn

class Indicator():
    def __init__(self,label,predict,nb_class):
        predict = predict.detach().cpu().numpy()
        label = label.detach().cpu().numpy()
        pred = np.argmax(predict, axis=1)
        label = nb_class * label
        label = label + pred
        count = np.bincount(label, minlength=nb_class ** 2)  # 0->1
        self.confusion_matrix = count.reshape(nb_class, nb_class)
        print(self.confusion_matrix)

    def Kappa(self):
        matrix_sum = np.sum(self.confusion_matrix)  # matrix_sum
        true_positive = np.diag(self.confusion_matrix).sum()  # TP
        p0 = true_positive / matrix_sum
        temp = 0
        for i in range(self.confusion_matrix.shape[0]):
            temp = temp + self.confusion_matrix[i, :].sum() * self.confusion_matrix[:, i].sum()
        p1 = temp / (matrix_sum * matrix_sum)
        kappa = (p0 - p1) / (1 - p1)
        return kappa

    def Over_Accuracy(self):
        OA = np.diag(self.confusion_matrix).sum() / (np.sum(self.confusion_matrix) + 1e-10)
        # print(np.diag(self.confusion_matrix).sum())
        # print( (np.sum(self.confusion_matrix)))
        return OA

    def Average_Accuracy(self):
        true_positive = np.diag(self.confusion_matrix)  # TP
        condition_positive = np.sum(self.confusion_matrix, axis=1)  # TP+FN
        R_per_class = true_positive / (condition_positive + 1e-10)  # TP/P
        AA = np.nanmean(R_per_class)
        return AA

if __name__ == '__main__':
     a = np.array([[1,2,3,4],[6,2,3,4],[2,2,6,4]])
     b = np.array([3,1,2])
     indic = Indicator(b,a,4)
     indic.Over_Accuracy()
























