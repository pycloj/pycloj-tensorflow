(ns tensorflow.contrib.training.RandomStrategy
  "Returns a random PS task for op placement.

  This may perform better than the default round-robin placement if you
  have a large number of variables. Depending on your architecture and
  number of parameter servers, round-robin can lead to situations where
  all of one type of variable is placed on a single PS task, which may
  lead to contention issues.

  This strategy uses a hash function on the name of each op for deterministic
  placement.
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce training (import-module "tensorflow.contrib.training"))
(defn RandomStrategy 
  "Returns a random PS task for op placement.

  This may perform better than the default round-robin placement if you
  have a large number of variables. Depending on your architecture and
  number of parameter servers, round-robin can lead to situations where
  all of one type of variable is placed on a single PS task, which may
  lead to contention issues.

  This strategy uses a hash function on the name of each op for deterministic
  placement.
  "
  [num_ps_tasks  & {:keys [seed]} ]
    (py/call-attr-kw training "RandomStrategy" [num_ps_tasks] {:seed seed }))
