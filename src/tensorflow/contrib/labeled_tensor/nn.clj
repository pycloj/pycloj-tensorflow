(ns tensorflow.contrib.labeled-tensor.python.ops.nn
  "Neural network ops for LabeledTensors."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce nn (import-module "tensorflow.contrib.labeled_tensor.python.ops.nn"))

(defn crelu 
  "LabeledTensor version of `tf.crelu`.

    See `tf.crelu` for full details.

    Args:
      labeled_tensor: Input tensor.
      name: Optional op name.

    Returns:
      A LabeledTensor with result of applying `tf.crelu` elementwise.
    "
  [ labeled_tensor name ]
  (py/call-attr nn "crelu"  labeled_tensor name ))

(defn elu 
  "LabeledTensor version of `tf.elu`.

    See `tf.elu` for full details.

    Args:
      labeled_tensor: Input tensor.
      name: Optional op name.

    Returns:
      A LabeledTensor with result of applying `tf.elu` elementwise.
    "
  [ labeled_tensor name ]
  (py/call-attr nn "elu"  labeled_tensor name ))

(defn l2-loss 
  "LabeledTensor version of `tf.l2_loss`.

    See `tf.l2_loss` for full details.

    Args:
      labeled_tensor: Input tensor.
      name: Optional op name.

    Returns:
      A LabeledTensor with result of applying `tf.l2_loss` elementwise.
    "
  [ labeled_tensor name ]
  (py/call-attr nn "l2_loss"  labeled_tensor name ))

(defn log-softmax 
  "LabeledTensor version of `tf.log_softmax`.

    See `tf.log_softmax` for full details.

    Args:
      labeled_tensor: Input tensor.
      name: Optional op name.

    Returns:
      A LabeledTensor with result of applying `tf.log_softmax` elementwise.
    "
  [ labeled_tensor name ]
  (py/call-attr nn "log_softmax"  labeled_tensor name ))

(defn relu 
  "LabeledTensor version of `tf.relu`.

    See `tf.relu` for full details.

    Args:
      labeled_tensor: Input tensor.
      name: Optional op name.

    Returns:
      A LabeledTensor with result of applying `tf.relu` elementwise.
    "
  [ labeled_tensor name ]
  (py/call-attr nn "relu"  labeled_tensor name ))

(defn relu6 
  "LabeledTensor version of `tf.relu6`.

    See `tf.relu6` for full details.

    Args:
      labeled_tensor: Input tensor.
      name: Optional op name.

    Returns:
      A LabeledTensor with result of applying `tf.relu6` elementwise.
    "
  [ labeled_tensor name ]
  (py/call-attr nn "relu6"  labeled_tensor name ))

(defn sigmoid-cross-entropy-with-logits 
  "LabeledTensor version of `tf.sigmoid_cross_entropy_with_logits` with label based alignment.

    See `tf.sigmoid_cross_entropy_with_logits` for full details.

    Args:
      labeled_tensor_0: Input tensor.
      labeled_tensor_1: Input tensor.
      name: Optional op name.

    Returns:
      A LabeledTensor with result of applying `tf.sigmoid_cross_entropy_with_logits` elementwise.
    "
  [ labeled_tensor_0 labeled_tensor_1 name ]
  (py/call-attr nn "sigmoid_cross_entropy_with_logits"  labeled_tensor_0 labeled_tensor_1 name ))

(defn softmax 
  "LabeledTensor version of `tf.softmax`.

    See `tf.softmax` for full details.

    Args:
      labeled_tensor: Input tensor.
      name: Optional op name.

    Returns:
      A LabeledTensor with result of applying `tf.softmax` elementwise.
    "
  [ labeled_tensor name ]
  (py/call-attr nn "softmax"  labeled_tensor name ))

(defn softmax-cross-entropy-with-logits 
  "LabeledTensor version of `tf.softmax_cross_entropy_with_logits` with label based alignment.

    See `tf.softmax_cross_entropy_with_logits` for full details.

    Args:
      labeled_tensor_0: Input tensor.
      labeled_tensor_1: Input tensor.
      name: Optional op name.

    Returns:
      A LabeledTensor with result of applying `tf.softmax_cross_entropy_with_logits` elementwise.
    "
  [ labeled_tensor_0 labeled_tensor_1 name ]
  (py/call-attr nn "softmax_cross_entropy_with_logits"  labeled_tensor_0 labeled_tensor_1 name ))

(defn softplus 
  "LabeledTensor version of `tf.softplus`.

    See `tf.softplus` for full details.

    Args:
      labeled_tensor: Input tensor.
      name: Optional op name.

    Returns:
      A LabeledTensor with result of applying `tf.softplus` elementwise.
    "
  [ labeled_tensor name ]
  (py/call-attr nn "softplus"  labeled_tensor name ))

(defn sparse-softmax-cross-entropy-with-logits 
  "LabeledTensor version of `tf.sparse_softmax_cross_entropy_with_logits` with label based alignment.

    See `tf.sparse_softmax_cross_entropy_with_logits` for full details.

    Args:
      labeled_tensor_0: Input tensor.
      labeled_tensor_1: Input tensor.
      name: Optional op name.

    Returns:
      A LabeledTensor with result of applying `tf.sparse_softmax_cross_entropy_with_logits` elementwise.
    "
  [ labeled_tensor_0 labeled_tensor_1 name ]
  (py/call-attr nn "sparse_softmax_cross_entropy_with_logits"  labeled_tensor_0 labeled_tensor_1 name ))
