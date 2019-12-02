(ns tensorflow.contrib.learn.python.learn.datasets.mnist
  "Functions for downloading and reading MNIST data (deprecated).

This module and all its submodules are deprecated. See
[contrib/learn/README.md](https://www.tensorflow.org/code/tensorflow/contrib/learn/README.md)
for migration instructions.
"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce mnist (import-module "tensorflow.contrib.learn.python.learn.datasets.mnist"))

(defn dense-to-one-hot 
  "Convert class labels from scalars to one-hot vectors. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Please use tf.one_hot on tensors."
  [ labels_dense num_classes ]
  (py/call-attr mnist "dense_to_one_hot"  labels_dense num_classes ))
(defn deprecated 
  "Decorator for marking functions or methods deprecated.

  This decorator logs a deprecation warning whenever the decorated function is
  called. It has the following format:

    <function> (from <module>) is deprecated and will be removed after <date>.
    Instructions for updating:
    <instructions>

  If `date` is None, 'after <date>' is replaced with 'in a future version'.
  <function> will include the class name if it is a method.

  It also edits the docstring of the function: ' (deprecated)' is appended
  to the first line of the docstring and a deprecation notice is prepended
  to the rest of the docstring.

  Args:
    date: String or None. The date the function is scheduled to be removed.
      Must be ISO 8601 (YYYY-MM-DD), or None.
    instructions: String. Instructions on how to update code using the
      deprecated function.
    warn_once: Boolean. Set to `True` to warn only the first time the decorated
      function is called. Otherwise, every call will log a warning.

  Returns:
    Decorated function or method.

  Raises:
    ValueError: If date is not None or in ISO 8601 format, or instructions are
      empty.
  "
  [date instructions  & {:keys [warn_once]} ]
    (py/call-attr-kw mnist "deprecated" [date instructions] {:warn_once warn_once }))

(defn extract-images 
  "Extract the images into a 4D uint8 numpy array [index, y, x, depth]. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Please use tf.data to implement this functionality.

Args:
  f: A file object that can be passed into a gzip reader.

Returns:
  data: A 4D uint8 numpy array [index, y, x, depth].

Raises:
  ValueError: If the bytestream does not start with 2051."
  [ f ]
  (py/call-attr mnist "extract_images"  f ))
(defn extract-labels 
  "Extract the labels into a 1D uint8 numpy array [index]. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Please use tf.data to implement this functionality.

Args:
  f: A file object that can be passed into a gzip reader.
  one_hot: Does one hot encoding for the result.
  num_classes: Number of classes for the one hot encoding.

Returns:
  labels: a 1D uint8 numpy array.

Raises:
  ValueError: If the bystream doesn't start with 2049."
  [f  & {:keys [one_hot num_classes]} ]
    (py/call-attr-kw mnist "extract_labels" [f] {:one_hot one_hot :num_classes num_classes }))

(defn load-mnist 
  "DEPRECATED FUNCTION

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Please use alternatives such as official/mnist/dataset.py from tensorflow/models."
  [ & {:keys [train_dir]} ]
   (py/call-attr-kw mnist "load_mnist" [] {:train_dir train_dir }))

(defn read-data-sets 
  "DEPRECATED FUNCTION

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Please use alternatives such as official/mnist/dataset.py from tensorflow/models."
  [train_dir & {:keys [fake_data one_hot dtype reshape validation_size seed source_url]
                       :or {seed None}} ]
    (py/call-attr-kw mnist "read_data_sets" [train_dir] {:fake_data fake_data :one_hot one_hot :dtype dtype :reshape reshape :validation_size validation_size :seed seed :source_url source_url }))
