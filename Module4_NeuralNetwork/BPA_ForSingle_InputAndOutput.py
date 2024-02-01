import numpy
import matplotlib.pyplot


def sigmod(sop):
    return 1.0/(1+numpy.exp(-1*sop))
def error(predicted, target):
    return numpy.power(predicted.target,2)
def error_predicted_deriv(predicted,target):
    return 2*(predicted -target)

def activation_sop_deriv(sop):
    return sigmod(sop)*(1.0-sigmod(sop))

def sop_w_deriv(x):
    return x
def update_w(w,grad,learning_rate):
    return w-learning_rate*grad