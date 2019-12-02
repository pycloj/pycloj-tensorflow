(ns tensorflow.contrib.slim.python.slim.data.tfexample-decoder.ItemHandler
  "Specifies the item-to-Features mapping for tf.parse_example.

  An ItemHandler both specifies a list of Features used for parsing an Example
  proto as well as a function that post-processes the results of Example
  parsing.
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

(defn ItemHandler 
  "Specifies the item-to-Features mapping for tf.parse_example.

  An ItemHandler both specifies a list of Features used for parsing an Example
  proto as well as a function that post-processes the results of Example
  parsing.
  "
  [ keys ]
  (py/call-attr tfexample-decoder "ItemHandler"  keys ))

(defn keys 
  ""
  [ self ]
    (py/call-attr self "keys"))

(defn tensors-to-item 
  "Maps the given dictionary of tensors to the requested item.

    Args:
      keys_to_tensors: a mapping of TF-Example keys to parsed tensors.

    Returns:
      the final tensor representing the item being handled.
    "
  [ self keys_to_tensors ]
  (py/call-attr self "tensors_to_item"  self keys_to_tensors ))
