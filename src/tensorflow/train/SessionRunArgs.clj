(ns tensorflow.train.SessionRunArgs
  "Represents arguments to be added to a `Session.run()` call.

  Args:
    fetches: Exactly like the 'fetches' argument to Session.Run().
      Can be a single tensor or op, a list of 'fetches' or a dictionary
      of fetches.  For example:
        fetches = global_step_tensor
        fetches = [train_op, summary_op, global_step_tensor]
        fetches = {'step': global_step_tensor, 'summ': summary_op}
      Note that this can recurse as expected:
        fetches = {'step': global_step_tensor,
                   'ops': [train_op, check_nan_op]}
    feed_dict: Exactly like the `feed_dict` argument to `Session.Run()`
    options: Exactly like the `options` argument to `Session.run()`, i.e., a
      config_pb2.RunOptions proto.
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

(defn SessionRunArgs 
  "Represents arguments to be added to a `Session.run()` call.

  Args:
    fetches: Exactly like the 'fetches' argument to Session.Run().
      Can be a single tensor or op, a list of 'fetches' or a dictionary
      of fetches.  For example:
        fetches = global_step_tensor
        fetches = [train_op, summary_op, global_step_tensor]
        fetches = {'step': global_step_tensor, 'summ': summary_op}
      Note that this can recurse as expected:
        fetches = {'step': global_step_tensor,
                   'ops': [train_op, check_nan_op]}
    feed_dict: Exactly like the `feed_dict` argument to `Session.Run()`
    options: Exactly like the `options` argument to `Session.run()`, i.e., a
      config_pb2.RunOptions proto.
  "
  [ fetches feed_dict options ]
  (py/call-attr train "SessionRunArgs"  fetches feed_dict options ))

(defn feed-dict 
  "Alias for field number 1"
  [ self ]
    (py/call-attr self "feed_dict"))

(defn fetches 
  "Alias for field number 0"
  [ self ]
    (py/call-attr self "fetches"))

(defn options 
  "Alias for field number 2"
  [ self ]
    (py/call-attr self "options"))
