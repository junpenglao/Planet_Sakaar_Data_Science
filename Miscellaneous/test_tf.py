import tensorflow as tf
import numpy as np

delta = .01
@autograph.convert()
def huber_loss(a):
  if tf.abs(a) <= delta:
    loss = a * a / 2
  else:
    loss = delta * (tf.abs(a) - delta / 2)
  return loss

with tf.Graph().as_default():
  x_tensor = tf.constant(9.0)

  # The converted function works like a regular op: tensors in, tensors out.
  huber_loss_tensor = huber_loss(x_tensor)

  with tf.Session() as sess:
    print('TensorFlow result: %2.2f\n' % sess.run(huber_loss_tensor))


