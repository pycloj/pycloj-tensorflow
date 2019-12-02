(ns tensorflow.-api.v1.compat.v1.saved-model.main-op
  "SavedModel main op.

Builds a main op that defines the sequence of ops to be run as part of the
SavedModel load/restore operations.

"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce main-op (import-module "tensorflow._api.v1.compat.v1.saved_model.main_op"))

(defn main-op 
  "Returns a main op to init variables and tables. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
This function will only be available through the v1 compatibility library as tf.compat.v1.saved_model.main_op.main_op.

Returns the main op including the group of ops that initializes all
variables, initializes local variables and initialize all tables.

Returns:
  The set of ops to be run as part of the main op upon the load operation."
  [  ]
  (py/call-attr main-op "main_op"  ))

(defn main-op-with-restore 
  "Returns a main op to init variables, tables and restore the graph. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
This function will only be available through the v1 compatibility library as tf.compat.v1.saved_model.main_op_with_restore or tf.compat.v1.saved_model.main_op.main_op_with_restore.

Returns the main op including the group of ops that initializes all
variables, initialize local variables, initialize all tables and the restore
op name.

Args:
  restore_op_name: Name of the op to use to restore the graph.

Returns:
  The set of ops to be run as part of the main op upon the load operation."
  [ restore_op_name ]
  (py/call-attr main-op "main_op_with_restore"  restore_op_name ))
