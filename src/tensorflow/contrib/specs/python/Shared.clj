(ns tensorflow.contrib.specs.python.Shared
  "Wraps a scope with variable reuse around the subnetwork.

  This function is still experimental.

  Attributes:
      f: The shared subnetwork.
      name: A name for the shared scope.
      used: A flag indicating whether the scope has already been used.
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce python (import-module "tensorflow.contrib.specs.python"))

(defn Shared 
  "Wraps a scope with variable reuse around the subnetwork.

  This function is still experimental.

  Attributes:
      f: The shared subnetwork.
      name: A name for the shared scope.
      used: A flag indicating whether the scope has already been used.
  "
  [ subnet name scope ]
  (py/call-attr specs "Shared"  subnet name scope ))

(defn funcall 
  "Apply the shared operator to an input.

    This wraps a variable scope around the creation of the subnet.

    Args:
        x: The input argument on which the subnet is invoked.

    Returns:
        The output tensor from invoking the subnet constructor.
    "
  [ self x ]
  (py/call-attr self "funcall"  self x ))
