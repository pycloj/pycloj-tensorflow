(ns tensorflow.contrib.slim.python.slim.data.data-provider.DataProvider
  "Maps a list of requested data items to tensors from a data source.

  All data providers must inherit from DataProvider and implement the Get
  method which returns arbitrary types of data. No assumption is made about the
  source of the data nor the mechanism for providing it.
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce data-provider (import-module "tensorflow.contrib.slim.python.slim.data.data_provider"))

(defn DataProvider 
  "Maps a list of requested data items to tensors from a data source.

  All data providers must inherit from DataProvider and implement the Get
  method which returns arbitrary types of data. No assumption is made about the
  source of the data nor the mechanism for providing it.
  "
  [ items_to_tensors num_samples ]
  (py/call-attr data-provider "DataProvider"  items_to_tensors num_samples ))

(defn get 
  "Returns a list of tensors specified by the given list of items.

    The list of items is arbitrary different data providers satisfy different
    lists of items. For example the Pascal VOC might accept items 'image' and
    'semantics', whereas the NYUDepthV2 data provider might accept items
    'image', 'depths' and 'normals'.

    Args:
      items: a list of strings, each of which indicate a particular data type.

    Returns:
      a list of tensors, whose length matches the length of `items`, where each
      tensor corresponds to each item.

    Raises:
      ValueError: if any of the items cannot be satisfied.
    "
  [ self items ]
  (py/call-attr self "get"  self items ))

(defn list-items 
  "Returns the list of item names that can be provided by the data provider.

    Returns:
      a list of item names that can be passed to Get([items]).
    "
  [ self  ]
  (py/call-attr self "list_items"  self  ))

(defn num-samples 
  "Returns the number of data samples in the dataset.

    Returns:
      a positive whole number.
    "
  [ self  ]
  (py/call-attr self "num_samples"  self  ))
