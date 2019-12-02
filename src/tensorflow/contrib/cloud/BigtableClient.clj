(ns tensorflow.contrib.cloud.BigtableClient
  "BigtableClient is the entrypoint for interacting with Cloud Bigtable in TF.

  BigtableClient encapsulates a connection to Cloud Bigtable, and exposes the
  `table` method to open a Bigtable table.
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce cloud (import-module "tensorflow.contrib.cloud"))

(defn BigtableClient 
  "BigtableClient is the entrypoint for interacting with Cloud Bigtable in TF.

  BigtableClient encapsulates a connection to Cloud Bigtable, and exposes the
  `table` method to open a Bigtable table.
  "
  [ project_id instance_id connection_pool_size max_receive_message_size ]
  (py/call-attr cloud "BigtableClient"  project_id instance_id connection_pool_size max_receive_message_size ))

(defn table 
  "Opens a table and returns a `tf.contrib.bigtable.BigtableTable` object.

    Args:
      name: A `tf.string` `tf.Tensor` name of the table to open.
      snapshot: Either a `tf.string` `tf.Tensor` snapshot id, or `True` to
        request the creation of a snapshot. (Note: currently unimplemented.)

    Returns:
      A `tf.contrib.bigtable.BigtableTable` Python object representing the
      operations available on the table.
    "
  [ self name snapshot ]
  (py/call-attr self "table"  self name snapshot ))
