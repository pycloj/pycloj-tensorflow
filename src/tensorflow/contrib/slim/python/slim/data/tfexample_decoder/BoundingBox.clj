(ns tensorflow.contrib.slim.python.slim.data.tfexample-decoder.BoundingBox
  "An ItemHandler that concatenates a set of parsed Tensors to Bounding Boxes.
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
(defn BoundingBox 
  "An ItemHandler that concatenates a set of parsed Tensors to Bounding Boxes.
  "
  [keys  & {:keys [prefix]} ]
    (py/call-attr-kw tfexample-decoder "BoundingBox" [keys] {:prefix prefix }))

(defn keys 
  ""
  [ self ]
    (py/call-attr self "keys"))

(defn tensors-to-item 
  "Maps the given dictionary of tensors to a concatenated list of bboxes.

    Args:
      keys_to_tensors: a mapping of TF-Example keys to parsed tensors.

    Returns:
      [num_boxes, 4] tensor of bounding box coordinates,
        i.e. 1 bounding box per row, in order [y_min, x_min, y_max, x_max].
    "
  [ self keys_to_tensors ]
  (py/call-attr self "tensors_to_item"  self keys_to_tensors ))
