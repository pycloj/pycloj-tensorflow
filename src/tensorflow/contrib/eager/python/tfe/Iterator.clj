(ns tensorflow.contrib.eager.python.tfe.Iterator
  "An iterator producing tf.Tensor objects from a tf.data.Dataset.

  NOTE: Unlike the iterator created by the
  `tf.data.Dataset.make_one_shot_iterator` method, this class enables
  additional experimental functionality, such as prefetching to the GPU.
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce tfe (import-module "tensorflow.contrib.eager.python.tfe"))

(defn Iterator 
  "An iterator producing tf.Tensor objects from a tf.data.Dataset.

  NOTE: Unlike the iterator created by the
  `tf.data.Dataset.make_one_shot_iterator` method, this class enables
  additional experimental functionality, such as prefetching to the GPU.
  "
  [ dataset ]
  (py/call-attr eager "Iterator"  dataset ))

(defn element-spec 
  "The type specification of an element of this iterator.

    Returns:
      A nested structure of `tf.TypeSpec` objects matching the structure of an
      element of this iterator and specifying the type of individual components.
    "
  [ self ]
    (py/call-attr self "element_spec"))

(defn get-next 
  "Returns a nested structure of `tf.Tensor`s containing the next element.

    Args:
      name: (Optional.) A name for the created operation. Currently unused.

    Returns:
      A nested structure of `tf.Tensor` objects.

    Raises:
      `tf.errors.OutOfRangeError`: If the end of the dataset has been reached.
    "
  [ self name ]
  (py/call-attr self "get_next"  self name ))

(defn next 
  "Returns a nested structure of `Tensor`s containing the next element."
  [ self  ]
  (py/call-attr self "next"  self  ))

(defn output-classes 
  "Returns the class of each component of an element of this iterator. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Use `tf.compat.v1.data.get_output_classes(iterator)`.

The expected values are `tf.Tensor` and `tf.SparseTensor`.

Returns:
  A nested structure of Python `type` objects corresponding to each
  component of an element of this dataset."
  [ self ]
    (py/call-attr self "output_classes"))

(defn output-shapes 
  "Returns the shape of each component of an element of this iterator. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Use `tf.compat.v1.data.get_output_shapes(iterator)`.

Returns:
  A nested structure of `tf.TensorShape` objects corresponding to each
  component of an element of this dataset."
  [ self ]
    (py/call-attr self "output_shapes"))

(defn output-types 
  "Returns the type of each component of an element of this iterator. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Use `tf.compat.v1.data.get_output_types(iterator)`.

Returns:
  A nested structure of `tf.DType` objects corresponding to each component
  of an element of this dataset."
  [ self ]
    (py/call-attr self "output_types"))
