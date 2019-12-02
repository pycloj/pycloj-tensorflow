(ns tensorflow.contrib.specs.python.specs-ops
  "Operators for concise TensorFlow network models.

This module is used as an environment for evaluating expressions
in the \"specs\" DSL.
"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce specs-ops (import-module "tensorflow.contrib.specs.python.specs_ops"))

(defn Dwm 
  "Depth-wise convolution + softmax (used after LSTM)."
  [ n ]
  (py/call-attr specs-ops "Dwm"  n ))

(defn Dws 
  "Depth-wise convolution + sigmoid (used after LSTM)."
  [ n ]
  (py/call-attr specs-ops "Dws"  n ))

(defn External 
  "Import a function from an external module.

  Note that the `module_name` must be a module name
  that works with the usual import mechanisms. Shorthands
  like \"tf.nn\" will not work.

  Args:
      module_name: name of the module
      function_name: name of the function within the module

  Returns:
      Function-wrapped value of symbol.
  "
  [ module_name function_name ]
  (py/call-attr specs-ops "External"  module_name function_name ))

(defn Import 
  "Import a function by exec.

  Args:
      statements: Python statements

  Returns:
      Function-wrapped value of `f`.

  Raises:
      ValueError: the statements didn't define a value for \"f\"
  "
  [ statements ]
  (py/call-attr specs-ops "Import"  statements ))

(defn Var 
  "Implements an operator that generates a variable.

  This function is still experimental. Use it only
  for generating a single variable instance for
  each name.

  Args:
      name: Name of the variable.
      *args: Other arguments to get_variable.
      **kw: Other keywords for get_variable.

  Returns:
      A specs object for generating a variable.
  "
  [ name ]
  (py/call-attr specs-ops "Var"  name ))

(defn debug 
  "Turn on/off debugging mode.

  Debugging mode prints more information about the construction
  of a network.

  Args:
      mode: True if turned on, False otherwise
  "
  [ & {:keys [mode]} ]
   (py/call-attr-kw specs-ops "debug" [] {:mode mode }))
