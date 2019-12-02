(ns tensorflow.-api.v1.compat.v1.data
  "`tf.data.Dataset` API for input pipelines.

See [Importing Data](https://tensorflow.org/guide/datasets) for an overview.

"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce data (import-module "tensorflow._api.v1.compat.v1.data"))

(defn get-output-classes 
  "Returns the output classes of a `Dataset` or `Iterator` elements.

  This utility method replaces the deprecated-in-V2
  `tf.compat.v1.Dataset.output_classes` property.

  Args:
    dataset_or_iterator: A `tf.data.Dataset` or `tf.data.IteratorV2`.

  Returns:
    A nested structure of Python `type` objects matching the structure of the
    dataset / iterator elements and specifying the class of the individual
    components.
  "
  [ dataset_or_iterator ]
  (py/call-attr data "get_output_classes"  dataset_or_iterator ))

(defn get-output-shapes 
  "Returns the output shapes of a `Dataset` or `Iterator` elements.

  This utility method replaces the deprecated-in-V2
  `tf.compat.v1.Dataset.output_shapes` property.

  Args:
    dataset_or_iterator: A `tf.data.Dataset` or `tf.data.Iterator`.

  Returns:
    A nested structure of `tf.TensorShape` objects matching the structure of
    the dataset / iterator elements and specifying the shape of the individual
    components.
  "
  [ dataset_or_iterator ]
  (py/call-attr data "get_output_shapes"  dataset_or_iterator ))

(defn get-output-types 
  "Returns the output shapes of a `Dataset` or `Iterator` elements.

  This utility method replaces the deprecated-in-V2
  `tf.compat.v1.Dataset.output_types` property.

  Args:
    dataset_or_iterator: A `tf.data.Dataset` or `tf.data.Iterator`.

  Returns:
    A nested structure of `tf.DType` objects objects matching the structure of
    dataset / iterator elements and specifying the shape of the individual
    components.
  "
  [ dataset_or_iterator ]
  (py/call-attr data "get_output_types"  dataset_or_iterator ))

(defn make-initializable-iterator 
  "Creates a `tf.compat.v1.data.Iterator` for enumerating the elements of a dataset.

  Note: The returned iterator will be in an uninitialized state,
  and you must run the `iterator.initializer` operation before using it:

  ```python
  dataset = ...
  iterator = tf.compat.v1.data.make_initializable_iterator(dataset)
  # ...
  sess.run(iterator.initializer)
  ```

  Args:
    dataset: A `tf.data.Dataset`.
    shared_name: (Optional.) If non-empty, the returned iterator will be shared
      under the given name across multiple sessions that share the same devices
      (e.g. when using a remote server).

  Returns:
    A `tf.compat.v1.data.Iterator` over the elements of `dataset`.

  Raises:
    RuntimeError: If eager execution is enabled.
  "
  [ dataset shared_name ]
  (py/call-attr data "make_initializable_iterator"  dataset shared_name ))

(defn make-one-shot-iterator 
  "Creates a `tf.compat.v1.data.Iterator` for enumerating the elements of a dataset.

  Note: The returned iterator will be initialized automatically.
  A \"one-shot\" iterator does not support re-initialization.

  Args:
    dataset: A `tf.data.Dataset`.

  Returns:
    A `tf.compat.v1.data.Iterator` over the elements of this dataset.
  "
  [ dataset ]
  (py/call-attr data "make_one_shot_iterator"  dataset ))
