(ns tensorflow.contrib.specs.python.specs-lib
  "Implement the \"specs\" DSL for describing deep networks."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce specs-lib (import-module "tensorflow.contrib.specs.python.specs_lib"))

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
  (py/call-attr specs-lib "External"  module_name function_name ))

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
  (py/call-attr specs-lib "Import"  statements ))

(defn check-keywords 
  "Check for common Python keywords in spec.

  This function discourages the use of complex constructs
  in TensorFlow specs; it doesn't completely prohibit them
  (if necessary, we could check the AST).

  Args:
      spec: spec string

  Raises:
      ValueError: raised if spec contains a prohibited keyword.
  "
  [ spec ]
  (py/call-attr specs-lib "check_keywords"  spec ))

(defn debug 
  "Turn on/off debugging mode.

  Debugging mode prints more information about the construction
  of a network.

  Args:
      mode: True if turned on, False otherwise
  "
  [ & {:keys [mode]} ]
   (py/call-attr-kw specs-lib "debug" [] {:mode mode }))
(defn get-positional 
  "Interpolates keyword arguments into argument lists.

  If `kw` contains keywords of the form \"_0\", \"_1\", etc., these
  are positionally interpolated into the argument list.

  Args:
      args: argument list
      kw: keyword dictionary
      kw_overrides: key/value pairs that override kw

  Returns:
      (new_args, new_kw), new argument lists and keyword dictionaries
      with values interpolated.
  "
  [args kw  & {:keys [kw_overrides]} ]
    (py/call-attr-kw specs-lib "get_positional" [args kw] {:kw_overrides kw_overrides }))
