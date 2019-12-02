(ns tensorflow.contrib.input-pipeline
  "Ops and modules related to input_pipeline.

@@obtain_next
"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce input-pipeline (import-module "tensorflow.contrib.input_pipeline"))

(defn obtain-next 
  "Basic wrapper for the ObtainNextOp.

  Args:
    string_list_tensor: A tensor that is a list of strings
    counter: an int64 ref tensor to keep track of which element is returned.

  Returns:
    An op that produces the element at counter + 1 in the list, round
    robin style.
  "
  [ string_list_tensor counter ]
  (py/call-attr input-pipeline "obtain_next"  string_list_tensor counter ))
