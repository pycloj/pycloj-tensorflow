(ns tensorflow.contrib.labeled-tensor.LabeledTensor
  "A tensor with annotated axes.

  It has the following invariants:
    1) The dimensionality of the tensor is equal to the number of elements
    in axes.
    2) The number of coordinate values in the ith dimension is equal to the
    size of the tensor in the ith dimension.

  Attributes:
    tensor: tf.Tensor containing the data.
    axes: lt.Axes containing axis names and coordinate labels.
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce labeled-tensor (import-module "tensorflow.contrib.labeled_tensor"))

(defn LabeledTensor 
  "A tensor with annotated axes.

  It has the following invariants:
    1) The dimensionality of the tensor is equal to the number of elements
    in axes.
    2) The number of coordinate values in the ith dimension is equal to the
    size of the tensor in the ith dimension.

  Attributes:
    tensor: tf.Tensor containing the data.
    axes: lt.Axes containing axis names and coordinate labels.
  "
  [ tensor axes ]
  (py/call-attr labeled-tensor "LabeledTensor"  tensor axes ))

(defn axes 
  ""
  [ self ]
    (py/call-attr self "axes"))

(defn dtype 
  ""
  [ self ]
    (py/call-attr self "dtype"))

(defn get-shape 
  "Returns the TensorShape that represents the shape of this tensor.

    See tf.Tensor.get_shape().

    Returns:
      A TensorShape representing the shape of this tensor.
    "
  [ self  ]
  (py/call-attr self "get_shape"  self  ))

(defn name 
  ""
  [ self ]
    (py/call-attr self "name"))

(defn shape 
  ""
  [ self ]
    (py/call-attr self "shape"))

(defn tensor 
  ""
  [ self ]
    (py/call-attr self "tensor"))
