(ns tensorflow.contrib.slim.python.slim.data.dataset-data-provider.DatasetDataProvider
  ""
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce dataset-data-provider (import-module "tensorflow.contrib.slim.python.slim.data.dataset_data_provider"))

(defn DatasetDataProvider 
  ""
  [dataset & {:keys [num_readers reader_kwargs shuffle num_epochs common_queue_capacity common_queue_min record_key seed scope]
                       :or {reader_kwargs None num_epochs None seed None scope None}} ]
    (py/call-attr-kw dataset-data-provider "DatasetDataProvider" [dataset] {:num_readers num_readers :reader_kwargs reader_kwargs :shuffle shuffle :num_epochs num_epochs :common_queue_capacity common_queue_capacity :common_queue_min common_queue_min :record_key record_key :seed seed :scope scope }))

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
