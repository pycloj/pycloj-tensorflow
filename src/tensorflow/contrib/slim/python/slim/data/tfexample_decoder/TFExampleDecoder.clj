(ns tensorflow.contrib.slim.python.slim.data.tfexample-decoder.TFExampleDecoder
  "A decoder for TensorFlow Examples.

  Decoding Example proto buffers is comprised of two stages: (1) Example parsing
  and (2) tensor manipulation.

  In the first stage, the tf.io.parse_example function is called with a list of
  FixedLenFeatures and SparseLenFeatures. These instances tell TF how to parse
  the example. The output of this stage is a set of tensors.

  In the second stage, the resulting tensors are manipulated to provide the
  requested 'item' tensors.

  To perform this decoding operation, an ExampleDecoder is given a list of
  ItemHandlers. Each ItemHandler indicates the set of features for stage 1 and
  contains the instructions for post_processing its tensors for stage 2.
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce tfexample-decoder (import-module "tensorflow.contrib.slim.python.slim.data.tfexample_decoder"))

(defn TFExampleDecoder 
  "A decoder for TensorFlow Examples.

  Decoding Example proto buffers is comprised of two stages: (1) Example parsing
  and (2) tensor manipulation.

  In the first stage, the tf.io.parse_example function is called with a list of
  FixedLenFeatures and SparseLenFeatures. These instances tell TF how to parse
  the example. The output of this stage is a set of tensors.

  In the second stage, the resulting tensors are manipulated to provide the
  requested 'item' tensors.

  To perform this decoding operation, an ExampleDecoder is given a list of
  ItemHandlers. Each ItemHandler indicates the set of features for stage 1 and
  contains the instructions for post_processing its tensors for stage 2.
  "
  [ keys_to_features items_to_handlers ]
  (py/call-attr tfexample-decoder "TFExampleDecoder"  keys_to_features items_to_handlers ))

(defn decode 
  "Decodes the given serialized TF-example.

    Args:
      serialized_example: a serialized TF-example tensor.
      items: the list of items to decode. These must be a subset of the item
        keys in self._items_to_handlers. If `items` is left as None, then all
        of the items in self._items_to_handlers are decoded.

    Returns:
      the decoded items, a list of tensor.
    "
  [ self serialized_example items ]
  (py/call-attr self "decode"  self serialized_example items ))

(defn list-items 
  "See base class."
  [ self  ]
  (py/call-attr self "list_items"  self  ))
