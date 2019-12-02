(ns tensorflow.contrib.keras.api.keras.utils.CustomObjectScope
  "Provides a scope that changes to `_GLOBAL_CUSTOM_OBJECTS` cannot escape.

  Code within a `with` statement will be able to access custom objects
  by name. Changes to global custom objects persist
  within the enclosing `with` statement. At end of the `with` statement,
  global custom objects are reverted to state
  at beginning of the `with` statement.

  Example:

  Consider a custom object `MyObject` (e.g. a class):

  ```python
      with CustomObjectScope({'MyObject':MyObject}):
          layer = Dense(..., kernel_regularizer='MyObject')
          # save, load, etc. will recognize custom object by name
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
(defonce utils (import-module "tensorflow.contrib.keras.api.keras.utils"))

(defn CustomObjectScope 
  "Provides a scope that changes to `_GLOBAL_CUSTOM_OBJECTS` cannot escape.

  Code within a `with` statement will be able to access custom objects
  by name. Changes to global custom objects persist
  within the enclosing `with` statement. At end of the `with` statement,
  global custom objects are reverted to state
  at beginning of the `with` statement.

  Example:

  Consider a custom object `MyObject` (e.g. a class):

  ```python
      with CustomObjectScope({'MyObject':MyObject}):
          layer = Dense(..., kernel_regularizer='MyObject')
          # save, load, etc. will recognize custom object by name
  ```
  "
  [  ]
  (py/call-attr utils "CustomObjectScope"  ))
