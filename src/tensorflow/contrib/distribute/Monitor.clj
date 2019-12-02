(ns tensorflow.contrib.distribute.Monitor
  "Executes training steps, recovers and checkpoints.

  Note that this class is particularly preliminary, experimental, and
  expected to change.
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce distribute (import-module "tensorflow.contrib.distribute"))

(defn Monitor 
  "Executes training steps, recovers and checkpoints.

  Note that this class is particularly preliminary, experimental, and
  expected to change.
  "
  [ step_callable session ]
  (py/call-attr distribute "Monitor"  step_callable session ))

(defn run-steps 
  ""
  [ self num_steps ]
  (py/call-attr self "run_steps"  self num_steps ))
