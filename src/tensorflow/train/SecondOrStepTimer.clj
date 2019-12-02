(ns tensorflow.train.SecondOrStepTimer
  "Timer that triggers at most once every N seconds or once every N steps.

  This symbol is also exported to v2 in tf.estimator namespace. See
  https://github.com/tensorflow/estimator/blob/master/tensorflow_estimator/python/estimator/hooks/basic_session_run_hooks.py
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

(defn SecondOrStepTimer 
  "Timer that triggers at most once every N seconds or once every N steps.

  This symbol is also exported to v2 in tf.estimator namespace. See
  https://github.com/tensorflow/estimator/blob/master/tensorflow_estimator/python/estimator/hooks/basic_session_run_hooks.py
  "
  [ every_secs every_steps ]
  (py/call-attr train "SecondOrStepTimer"  every_secs every_steps ))

(defn last-triggered-step 
  ""
  [ self  ]
  (py/call-attr self "last_triggered_step"  self  ))

(defn reset 
  ""
  [ self  ]
  (py/call-attr self "reset"  self  ))

(defn should-trigger-for-step 
  "Return true if the timer should trigger for the specified step.

    Args:
      step: Training step to trigger on.

    Returns:
      True if the difference between the current time and the time of the last
      trigger exceeds `every_secs`, or if the difference between the current
      step and the last triggered step exceeds `every_steps`. False otherwise.
    "
  [ self step ]
  (py/call-attr self "should_trigger_for_step"  self step ))

(defn update-last-triggered-step 
  ""
  [ self step ]
  (py/call-attr self "update_last_triggered_step"  self step ))
