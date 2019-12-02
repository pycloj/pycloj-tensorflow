(ns tensorflow.contrib.slim.python.slim.data.data-decoder.DataDecoder
  "An abstract class which is used to decode data for a provider."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce data-decoder (import-module "tensorflow.contrib.slim.python.slim.data.data_decoder"))

(defn DataDecoder 
  "An abstract class which is used to decode data for a provider."
  [  ]
  (py/call-attr data-decoder "DataDecoder"  ))

(defn decode 
  "Decodes the data to returns the tensors specified by the list of items.

    Args:
      data: A possibly encoded data format.
      items: A list of strings, each of which indicate a particular data type.

    Returns:
      A list of `Tensors`, whose length matches the length of `items`, where
      each `Tensor` corresponds to each item.

    Raises:
      ValueError: If any of the items cannot be satisfied.
    "
  [ self data items ]
  (py/call-attr self "decode"  self data items ))

(defn list-items 
  "Lists the names of the items that the decoder can decode.

    Returns:
      A list of string names.
    "
  [ self  ]
  (py/call-attr self "list_items"  self  ))
