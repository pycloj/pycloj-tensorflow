(ns tensorflow.-api.v1.compat.v1.IndexedSlices
  "A sparse representation of a set of tensor slices at given indices.

  This class is a simple wrapper for a pair of `Tensor` objects:

  * `values`: A `Tensor` of any dtype with shape `[D0, D1, ..., Dn]`.
  * `indices`: A 1-D integer `Tensor` with shape `[D0]`.

  An `IndexedSlices` is typically used to represent a subset of a larger
  tensor `dense` of shape `[LARGE0, D1, .. , DN]` where `LARGE0 >> D0`.
  The values in `indices` are the indices in the first dimension of
  the slices that have been extracted from the larger tensor.

  The dense tensor `dense` represented by an `IndexedSlices` `slices` has

  ```python
  dense[slices.indices[i], :, :, :, ...] = slices.values[i, :, :, :, ...]
  ```

  The `IndexedSlices` class is used principally in the definition of
  gradients for operations that have sparse gradients
  (e.g. `tf.gather`).

  Contrast this representation with
  `tf.SparseTensor`,
  which uses multi-dimensional indices and scalar values.
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce v1 (import-module "tensorflow._api.v1.compat.v1"))

(defn IndexedSlices 
  "A sparse representation of a set of tensor slices at given indices.

  This class is a simple wrapper for a pair of `Tensor` objects:

  * `values`: A `Tensor` of any dtype with shape `[D0, D1, ..., Dn]`.
  * `indices`: A 1-D integer `Tensor` with shape `[D0]`.

  An `IndexedSlices` is typically used to represent a subset of a larger
  tensor `dense` of shape `[LARGE0, D1, .. , DN]` where `LARGE0 >> D0`.
  The values in `indices` are the indices in the first dimension of
  the slices that have been extracted from the larger tensor.

  The dense tensor `dense` represented by an `IndexedSlices` `slices` has

  ```python
  dense[slices.indices[i], :, :, :, ...] = slices.values[i, :, :, :, ...]
  ```

  The `IndexedSlices` class is used principally in the definition of
  gradients for operations that have sparse gradients
  (e.g. `tf.gather`).

  Contrast this representation with
  `tf.SparseTensor`,
  which uses multi-dimensional indices and scalar values.
  "
  [ values indices dense_shape ]
  (py/call-attr v1 "IndexedSlices"  values indices dense_shape ))

(defn consumers 
  ""
  [ self  ]
  (py/call-attr self "consumers"  self  ))

(defn dense-shape 
  "A 1-D `Tensor` containing the shape of the corresponding dense tensor."
  [ self ]
    (py/call-attr self "dense_shape"))

(defn device 
  "The name of the device on which `values` will be produced, or `None`."
  [ self ]
    (py/call-attr self "device"))

(defn dtype 
  "The `DType` of elements in this tensor."
  [ self ]
    (py/call-attr self "dtype"))

(defn graph 
  "The `Graph` that contains the values, indices, and shape tensors."
  [ self ]
    (py/call-attr self "graph"))

(defn indices 
  "A 1-D `Tensor` containing the indices of the slices."
  [ self ]
    (py/call-attr self "indices"))

(defn name 
  "The name of this `IndexedSlices`."
  [ self ]
    (py/call-attr self "name"))

(defn op 
  "The `Operation` that produces `values` as an output."
  [ self ]
    (py/call-attr self "op"))

(defn values 
  "A `Tensor` containing the values of the slices."
  [ self ]
    (py/call-attr self "values"))
