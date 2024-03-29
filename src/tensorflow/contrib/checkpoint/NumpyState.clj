(ns tensorflow.contrib.checkpoint.NumpyState
  "A trackable object whose NumPy array attributes are saved/restored.

  Example usage:

  ```python
  arrays = tf.contrib.checkpoint.NumpyState()
  checkpoint = tf.train.Checkpoint(numpy_arrays=arrays)
  arrays.x = numpy.zeros([3, 4])
  save_path = checkpoint.save(\"/tmp/ckpt\")
  arrays.x[1, 1] = 4.
  checkpoint.restore(save_path)
  assert (arrays.x == numpy.zeros([3, 4])).all()

  second_checkpoint = tf.train.Checkpoint(
      numpy_arrays=tf.contrib.checkpoint.NumpyState())
  # Attributes of NumpyState objects are created automatically by restore()
  second_checkpoint.restore(save_path)
  assert (second_checkpoint.numpy_arrays.x == numpy.zeros([3, 4])).all()
  ```

  Note that `NumpyState` objects re-create the attributes of the previously
  saved object on `restore()`. This is in contrast to TensorFlow variables, for
  which a `Variable` object must be created and assigned to an attribute.

  This snippet works both when graph building and when executing eagerly. On
  save, the NumPy array(s) are fed as strings to be saved in the checkpoint (via
  a placeholder when graph building, or as a string constant when executing
  eagerly). When restoring they skip the TensorFlow graph entirely, and so no
  restore ops need be run. This means that restoration always happens eagerly,
  rather than waiting for `checkpoint.restore(...).run_restore_ops()` like
  TensorFlow variables when graph building.
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce checkpoint (import-module "tensorflow.contrib.checkpoint"))

(defn NumpyState 
  "A trackable object whose NumPy array attributes are saved/restored.

  Example usage:

  ```python
  arrays = tf.contrib.checkpoint.NumpyState()
  checkpoint = tf.train.Checkpoint(numpy_arrays=arrays)
  arrays.x = numpy.zeros([3, 4])
  save_path = checkpoint.save(\"/tmp/ckpt\")
  arrays.x[1, 1] = 4.
  checkpoint.restore(save_path)
  assert (arrays.x == numpy.zeros([3, 4])).all()

  second_checkpoint = tf.train.Checkpoint(
      numpy_arrays=tf.contrib.checkpoint.NumpyState())
  # Attributes of NumpyState objects are created automatically by restore()
  second_checkpoint.restore(save_path)
  assert (second_checkpoint.numpy_arrays.x == numpy.zeros([3, 4])).all()
  ```

  Note that `NumpyState` objects re-create the attributes of the previously
  saved object on `restore()`. This is in contrast to TensorFlow variables, for
  which a `Variable` object must be created and assigned to an attribute.

  This snippet works both when graph building and when executing eagerly. On
  save, the NumPy array(s) are fed as strings to be saved in the checkpoint (via
  a placeholder when graph building, or as a string constant when executing
  eagerly). When restoring they skip the TensorFlow graph entirely, and so no
  restore ops need be run. This means that restoration always happens eagerly,
  rather than waiting for `checkpoint.restore(...).run_restore_ops()` like
  TensorFlow variables when graph building.
  "
  [  ]
  (py/call-attr checkpoint "NumpyState"  ))
