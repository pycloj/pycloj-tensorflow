(ns tensorflow-estimator.python.estimator.api.-v1.estimator.EvalSpec
  "Configuration for the \"eval\" part for the `train_and_evaluate` call.

  `EvalSpec` combines details of evaluation of the trained model as well as its
  export. Evaluation consists of computing metrics to judge the performance of
  the trained model.  Export writes out the trained model on to external
  storage.
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce estimator (import-module "tensorflow_estimator.python.estimator.api._v1.estimator"))

(defn EvalSpec 
  "Configuration for the \"eval\" part for the `train_and_evaluate` call.

  `EvalSpec` combines details of evaluation of the trained model as well as its
  export. Evaluation consists of computing metrics to judge the performance of
  the trained model.  Export writes out the trained model on to external
  storage.
  "
  [input_fn & {:keys [steps name hooks exporters start_delay_secs throttle_secs]
                       :or {name None hooks None exporters None}} ]
    (py/call-attr-kw estimator "EvalSpec" [input_fn] {:steps steps :name name :hooks hooks :exporters exporters :start_delay_secs start_delay_secs :throttle_secs throttle_secs }))

(defn exporters 
  "Alias for field number 4"
  [ self ]
    (py/call-attr self "exporters"))

(defn hooks 
  "Alias for field number 3"
  [ self ]
    (py/call-attr self "hooks"))

(defn input-fn 
  "Alias for field number 0"
  [ self ]
    (py/call-attr self "input_fn"))

(defn name 
  "Alias for field number 2"
  [ self ]
    (py/call-attr self "name"))

(defn start-delay-secs 
  "Alias for field number 5"
  [ self ]
    (py/call-attr self "start_delay_secs"))

(defn steps 
  "Alias for field number 1"
  [ self ]
    (py/call-attr self "steps"))

(defn throttle-secs 
  "Alias for field number 6"
  [ self ]
    (py/call-attr self "throttle_secs"))
