
from __future__ import division, print_function
import tensorflow as tf


def averageGradients(tower_grads):
    average_grads = []
    # each grad_and_vars looks like ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    for grad_and_vars in zip(*tower_grads):
        grads=[tf.expand_dims(g, 0) for g,_ in grad_and_vars if g is not None]
        assert len(grads)>0, 'No variables have gradients'
        
        grad=tf.concat(grads,0)
        grad=tf.reduce_mean(grad, 0) 

        # variables are shared across towers, need only return first tower's variable refs
        average_grads.append((grad,grad_and_vars[0][1]))
    return average_grads


def binaryMaskDiceLoss(logits, labels, smooth=1e-5):
    axis = list(range(1, logits.shape.ndims - 1))
    logits=tf.cast(logits,tf.float32)
    labels=tf.cast(labels,tf.float32)

    probs = tf.nn.sigmoid(logits)[..., 0]
    label_sum = tf.reduce_sum(labels, axis=axis, name='label_sum')
    pred_sum = tf.reduce_sum(probs, axis=axis, name='pred_sum')
    intersection = tf.reduce_sum(labels * probs, axis=axis, name='intersection')
    sums=label_sum + pred_sum

    dice = tf.reduce_mean((2.0 * intersection + smooth) / (sums + smooth))
    return 1.0-dice

