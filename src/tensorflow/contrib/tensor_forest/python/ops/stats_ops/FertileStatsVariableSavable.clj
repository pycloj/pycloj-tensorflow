(ns tensorflow.contrib.tensor-forest.python.ops.stats-ops.FertileStatsVariableSavable
  "SaveableObject implementation for FertileStatsVariable."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce stats-ops (import-module "tensorflow.contrib.tensor_forest.python.ops.stats_ops"))

(defn FertileStatsVariableSavable 
  "SaveableObject implementation for FertileStatsVariable."
  [ params stats_handle create_op name ]
  (py/call-attr stats-ops "FertileStatsVariableSavable"  params stats_handle create_op name ))

(defn device 
  "The device for SaveSpec Tensors."
  [ self ]
    (py/call-attr self "device"))

(defn optional-restore 
  "A hint to restore assertions that this object is optional."
  [ self ]
    (py/call-attr self "optional_restore"))

(defn restore 
  "Restores the associated tree from 'restored_tensors'.

    Args:
      restored_tensors: the tensors that were loaded from a checkpoint.
      unused_restored_shapes: the shapes this object should conform to after
        restore. Not meaningful for trees.

    Returns:
      The operation that restores the state of the tree variable.
    "
  [ self restored_tensors unused_restored_shapes ]
  (py/call-attr self "restore"  self restored_tensors unused_restored_shapes ))
