(ns tensorflow.train.Scaffold
  "Structure to create or gather pieces commonly needed to train a model.

  When you build a model for training you usually need ops to initialize
  variables, a `Saver` to checkpoint them, an op to collect summaries for
  the visualizer, and so on.

  Various libraries built on top of the core TensorFlow library take care of
  creating some or all of these pieces and storing them in well known
  collections in the graph.  The `Scaffold` class helps pick these pieces from
  the graph collections, creating and adding them to the collections if needed.

  If you call the scaffold constructor without any arguments, it will pick
  pieces from the collections, creating default ones if needed when
  `scaffold.finalize()` is called.  You can pass arguments to the constructor to
  provide your own pieces.  Pieces that you pass to the constructor are not
  added to the graph collections.

  The following pieces are directly accessible as attributes of the `Scaffold`
  object:

  * `saver`: A `tf.compat.v1.train.Saver` object taking care of saving the
  variables.
    Picked from and stored into the `SAVERS` collection in the graph by default.
  * `init_op`: An op to run to initialize the variables.  Picked from and
    stored into the `INIT_OP` collection in the graph by default.
  * `ready_op`: An op to verify that the variables are initialized.  Picked
    from and stored into the `READY_OP` collection in the graph by default.
  * `ready_for_local_init_op`: An op to verify that global state has been
    initialized and it is alright to run `local_init_op`.  Picked from and
    stored into the `READY_FOR_LOCAL_INIT_OP` collection in the graph by
    default. This is needed when the initialization of local variables depends
    on the values of global variables.
  * `local_init_op`: An op to initialize the local variables.  Picked
    from and stored into the `LOCAL_INIT_OP` collection in the graph by default.
  * `summary_op`: An op to run and merge the summaries in the graph.  Picked
    from and stored into the `SUMMARY_OP` collection in the graph by default.

  You can also pass the following additional pieces to the constructor:

  * `init_feed_dict`: A session feed dictionary that should be used when
     running the init op.
  * `init_fn`: A callable to run after the init op to perform additional
    initializations.  The callable will be called as
    `init_fn(scaffold, session)`.

  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce train (import-module "tensorflow.train"))

(defn Scaffold 
  "Structure to create or gather pieces commonly needed to train a model.

  When you build a model for training you usually need ops to initialize
  variables, a `Saver` to checkpoint them, an op to collect summaries for
  the visualizer, and so on.

  Various libraries built on top of the core TensorFlow library take care of
  creating some or all of these pieces and storing them in well known
  collections in the graph.  The `Scaffold` class helps pick these pieces from
  the graph collections, creating and adding them to the collections if needed.

  If you call the scaffold constructor without any arguments, it will pick
  pieces from the collections, creating default ones if needed when
  `scaffold.finalize()` is called.  You can pass arguments to the constructor to
  provide your own pieces.  Pieces that you pass to the constructor are not
  added to the graph collections.

  The following pieces are directly accessible as attributes of the `Scaffold`
  object:

  * `saver`: A `tf.compat.v1.train.Saver` object taking care of saving the
  variables.
    Picked from and stored into the `SAVERS` collection in the graph by default.
  * `init_op`: An op to run to initialize the variables.  Picked from and
    stored into the `INIT_OP` collection in the graph by default.
  * `ready_op`: An op to verify that the variables are initialized.  Picked
    from and stored into the `READY_OP` collection in the graph by default.
  * `ready_for_local_init_op`: An op to verify that global state has been
    initialized and it is alright to run `local_init_op`.  Picked from and
    stored into the `READY_FOR_LOCAL_INIT_OP` collection in the graph by
    default. This is needed when the initialization of local variables depends
    on the values of global variables.
  * `local_init_op`: An op to initialize the local variables.  Picked
    from and stored into the `LOCAL_INIT_OP` collection in the graph by default.
  * `summary_op`: An op to run and merge the summaries in the graph.  Picked
    from and stored into the `SUMMARY_OP` collection in the graph by default.

  You can also pass the following additional pieces to the constructor:

  * `init_feed_dict`: A session feed dictionary that should be used when
     running the init op.
  * `init_fn`: A callable to run after the init op to perform additional
    initializations.  The callable will be called as
    `init_fn(scaffold, session)`.

  "
  [ init_op init_feed_dict init_fn ready_op ready_for_local_init_op local_init_op summary_op saver copy_from_scaffold ]
  (py/call-attr train "Scaffold"  init_op init_feed_dict init_fn ready_op ready_for_local_init_op local_init_op summary_op saver copy_from_scaffold ))

(defn default-local-init-op 
  "Returns an op that groups the default local init ops.

    This op is used during session initialization when a Scaffold is
    initialized without specifying the local_init_op arg. It includes
    `tf.compat.v1.local_variables_initializer`,
    `tf.compat.v1.tables_initializer`, and also
    initializes local session resources.

    Returns:
      The default Scaffold local init op.
    "
  [ self  ]
  (py/call-attr self "default_local_init_op"  self  ))

(defn finalize 
  "Creates operations if needed and finalizes the graph."
  [ self  ]
  (py/call-attr self "finalize"  self  ))

(defn get-or-default 
  "Get from cache or create a default operation."
  [ self arg_name collection_key default_constructor ]
  (py/call-attr self "get_or_default"  self arg_name collection_key default_constructor ))

(defn init-feed-dict 
  ""
  [ self ]
    (py/call-attr self "init_feed_dict"))

(defn init-fn 
  ""
  [ self ]
    (py/call-attr self "init_fn"))

(defn init-op 
  ""
  [ self ]
    (py/call-attr self "init_op"))

(defn local-init-op 
  ""
  [ self ]
    (py/call-attr self "local_init_op"))

(defn ready-for-local-init-op 
  ""
  [ self ]
    (py/call-attr self "ready_for_local_init_op"))

(defn ready-op 
  ""
  [ self ]
    (py/call-attr self "ready_op"))

(defn saver 
  ""
  [ self ]
    (py/call-attr self "saver"))

(defn summary-op 
  ""
  [ self ]
    (py/call-attr self "summary_op"))
