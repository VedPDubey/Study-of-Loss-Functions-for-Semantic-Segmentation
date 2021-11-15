import tensorflow as tf
import keras.backend as K
from keras.losses import binary_crossentropy
import numpy as np
from sklearn.utils.extmath import cartesian
import math

ce_w = 0.5
ce_d_w = 0.5
e = K.epsilon()
smooth = 1e-6

class loss_functions(object):
    def __init__(self):
        print ("Loss Functions Initialized")

    def get_iou_vector(A, B):
    # Numpy version
        batch_size = A.shape[0]
        metric = 0.0
        for batch in range(batch_size):
            t, p = A[batch], B[batch]
            true = np.sum(t)
            pred = np.sum(p)
            
            # deal with empty mask first
            if true == 0:
                metric += (pred == 0)
                continue
            
            # non empty mask case.  Union is never empty 
            # hence it is safe to divide by its number of pixels
            intersection = np.sum(t * p)
            union = true + pred - intersection
            iou = intersection / union
            
            # iou metrric is a stepwise approximation of the real iou over 0.5
            iou = np.floor(max(0, (iou - 0.45)*20)) / 10
            
            metric += iou
            
        # teake the average over all images in batch
        metric /= batch_size
        return metric

    def IoU(label, pred):
        # Tensorflow version
        return tf.py_func(get_iou_vector, [label, pred > 0.5], tf.float64)

    def IoULoss(targets, inputs):
        smooth=1e-6
        #flatten label and prediction tensors
        inputs = keras.flatten(inputs)
        targets = keras.flatten(targets)
        
        intersection = keras.sum(targets*inputs)
        total = keras.sum(targets) + keras.sum(inputs)
        union = total - intersection
        
        IoU = (intersection + smooth) / (union + smooth)
        return 1 - IoU


    def FocalLoss(targets, inputs, alpha=0.8, gamma=2):    
        
        inputs = K.flatten(inputs)
        targets = K.flatten(targets)
        
        BCE = K.binary_crossentropy(targets, inputs)
        BCE_EXP = K.exp(-BCE)
        focal_loss = K.mean(alpha * K.pow((1-BCE_EXP), gamma) * BCE)
        
        return focal_loss


    def TverskyLoss(targets, inputs, alpha=0.5, beta=0.5, smooth=1e-6):
            
            #flatten label and prediction tensors
            inputs = K.flatten(inputs)
            targets = K.flatten(targets)
            
            #True Positives, False Positives & False Negatives
            TP = K.sum((inputs * targets))
            FP = K.sum(((1-targets) * inputs))
            FN = K.sum((targets * (1-inputs)))
        
            Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)  
            
            return 1 - Tversky
        

    def FocalTverskyLoss(targets, inputs, alpha=0.5, beta=0.5, gamma=1, smooth=1e-6):
        
            #flatten label and prediction tensors
            inputs = K.flatten(inputs)
            targets = K.flatten(targets)
            
            #True Positives, False Positives & False Negatives
            TP = K.sum((inputs * targets))
            FP = K.sum(((1-targets) * inputs))
            FN = K.sum((targets * (1-inputs)))
                
            Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)  
            FocalTversky = K.pow((1 - Tversky), gamma)
            
            return FocalTversky

    def dice_coef(y_true, y_pred):
        y_true_f = keras.flatten(y_true)
        y_pred_f = keras.flatten(y_pred)
        intersection = keras.sum(y_true_f * y_pred_f)
        return (2. * intersection + 1) / (keras.sum(y_true_f) + keras.sum(y_pred_f) + 1)

    def dice_coef_loss(y_true, y_pred):
        return 1-dice_coef(y_true, y_pred)

    ce_w = 0.5
    ce_d_w = 0.5
    e = K.epsilon()
    smooth = 1e-6

    def Combo_loss(y_true, y_pred):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        d = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
        y_pred_f = K.clip(y_pred_f, e, 1.0 - e)
        out = - (ce_w * y_true_f * K.log(y_pred_f)) + ((1 - ce_w) * (1.0 - y_true_f) * K.log(1.0 - y_pred_f))
        weighted_ce = K.mean(out, axis=-1)
        combo = (ce_d_w * weighted_ce) - ((1 - ce_d_w) * d)
        return 1-combo