(ns tensorflow.-api.v1.compat.v2.train.experimental.PythonState
  "A mixin for putting Python state in an object-based checkpoint.

  This is an abstract class which allows extensions to TensorFlow's object-based
  checkpointing (see `tf.train.Checkpoint`). For example a wrapper for NumPy
  arrays:

  ```python
  import io
  import numpy

  class NumpyWrapper(tf.train.experimental.PythonState):

    def __init__(self, array):
      self.array = array

    def serialize(self):
      string_file = io.BytesIO()
      try:
        numpy.save(string_file, self.array, allow_pickle=False)
        serialized = string_file.getvalue()
      finally:
        string_file.close()
      return serialized

    def deserialize(self, string_value):
      string_file = io.BytesIO(string_value)
      try:
        self.array = numpy.load(string_file, allow_pickle=False)
      finally:
        string_file.close()
  ```

  Instances of `NumpyWrapper` are checkpointable objects, and will be saved and
  restored from checkpoints along with TensorFlow state like variables.

  ```python
  root = tf.train.Checkpoint(numpy=NumpyWrapper(numpy.array([1.])))
  save_path = root.save(prefix)
  root.numpy.array *= 2.
  assert [2.] == root.numpy.array
  root.restore(save_path)
  assert [1.] == root.numpy.array
  ```
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce experimental (import-module "tensorflow._api.v1.compat.v2.train.experimental"))

(defn PythonState 
  "A mixin for putting Python state in an object-based checkpoint.

  This is an abstract class which allows extensions to TensorFlow's object-based
  checkpointing (see `tf.train.Checkpoint`). For example a wrapper for NumPy
  arrays:

  ```python
  import io
  import numpy

  class NumpyWrapper(tf.train.experimental.PythonState):

    def __init__(self, array):
      self.array = array

    def serialize(self):
      string_file = io.BytesIO()
      try:
        numpy.save(string_file, self.array, allow_pickle=False)
        serialized = string_file.getvalue()
      finally:
        string_file.close()
      return serialized

    def deserialize(self, string_value):
      string_file = io.BytesIO(string_value)
      try:
        self.array = numpy.load(string_file, allow_pickle=False)
      finally:
        string_file.close()
  ```

  Instances of `NumpyWrapper` are checkpointable objects, and will be saved and
  restored from checkpoints along with TensorFlow state like variables.

  ```python
  root = tf.train.Checkpoint(numpy=NumpyWrapper(numpy.array([1.])))
  save_path = root.save(prefix)
  root.numpy.array *= 2.
  assert [2.] == root.numpy.array
  root.restore(save_path)
  assert [1.] == root.numpy.array
  ```
  "
  [  ]
  (py/call-attr experimental "PythonState"  ))

(defn deserialize 
  "Callback to deserialize the object."
  [ self string_value ]
  (py/call-attr self "deserialize"  self string_value ))

(defn serialize 
  "Callback to serialize the object. Returns a string."
  [ self  ]
  (py/call-attr self "serialize"  self  ))
