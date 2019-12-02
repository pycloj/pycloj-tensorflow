(ns tensorflow.contrib.eager.python.tfe.EagerVariableStore
  "Wrapper allowing functional layers to be used with eager execution.

  When eager execution is enabled Variables get deleted when they go out of
  scope, and are not stored in global collections by default. A lot of code
  (mostly the functional layers in tf.layers) assumes that variables are kept in
  a global list.

  EagerVariableStore can be used in conjunction with this code to make it
  eager-friendly. For example, to create a dense layer, use:

  ```
    container = tfe.EagerVariableStore()
    for input in dataset_iterator:
      with container.as_default():
        x = tf.compat.v1.layers.dense(input, name=\"l1\")
    print(container.variables)  # Should print the variables used in the layer.
  ```
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce tfe (import-module "tensorflow.contrib.eager.python.tfe"))

(defn EagerVariableStore 
  "Wrapper allowing functional layers to be used with eager execution.

  When eager execution is enabled Variables get deleted when they go out of
  scope, and are not stored in global collections by default. A lot of code
  (mostly the functional layers in tf.layers) assumes that variables are kept in
  a global list.

  EagerVariableStore can be used in conjunction with this code to make it
  eager-friendly. For example, to create a dense layer, use:

  ```
    container = tfe.EagerVariableStore()
    for input in dataset_iterator:
      with container.as_default():
        x = tf.compat.v1.layers.dense(input, name=\"l1\")
    print(container.variables)  # Should print the variables used in the layer.
  ```
  "
  [ store ]
  (py/call-attr eager "EagerVariableStore"  store ))

(defn as-default 
  ""
  [ self  ]
  (py/call-attr self "as_default"  self  ))

(defn copy 
  "Copy this variable store and all of its contents.

    Variables contained in this store will be copied over to the new variable
    store, meaning that they can be modified without affecting the variables in
    this store.

    Returns:
      A new EagerVariableStore instance containing copied variables.
    "
  [ self  ]
  (py/call-attr self "copy"  self  ))

(defn non-trainable-variables 
  ""
  [ self  ]
  (py/call-attr self "non_trainable_variables"  self  ))

(defn trainable-variables 
  ""
  [ self  ]
  (py/call-attr self "trainable_variables"  self  ))

(defn variables 
  ""
  [ self  ]
  (py/call-attr self "variables"  self  ))
