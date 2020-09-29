import tensorflow as tf
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

latest_ckp = tf.train.latest_checkpoint('./')
print_tensors_in_checkpoint_file("/tmp/model.ckpt", all_tensors=True, tensor_name='')