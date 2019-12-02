(ns tensorflow.contrib.solvers.python.ops.util
  "Utility functions for solvers."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce util (import-module "tensorflow.contrib.solvers.python.ops.util"))

(defn create-operator 
  "Creates a linear operator from a rank-2 tensor."
  [ matrix ]
  (py/call-attr util "create_operator"  matrix ))

(defn dot 
  ""
  [ x y ]
  (py/call-attr util "dot"  x y ))

(defn identity-operator 
  "Creates a linear operator from a rank-2 identity tensor."
  [ matrix ]
  (py/call-attr util "identity_operator"  matrix ))

(defn l2norm 
  ""
  [ v ]
  (py/call-attr util "l2norm"  v ))

(defn l2norm-squared 
  ""
  [ v ]
  (py/call-attr util "l2norm_squared"  v ))

(defn l2normalize 
  ""
  [ v ]
  (py/call-attr util "l2normalize"  v ))
