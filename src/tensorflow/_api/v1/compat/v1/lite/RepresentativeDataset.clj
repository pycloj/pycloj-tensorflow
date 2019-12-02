(ns tensorflow.-api.v1.compat.v1.lite.RepresentativeDataset
  "Representative dataset to evaluate optimizations.

  A representative dataset that can be used to evaluate optimizations by the
  converter. E.g. converter can use these examples to estimate (min, max) ranges
  by calibrating the model on inputs. This can allow converter to quantize a
  converted floating point model.
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce lite (import-module "tensorflow._api.v1.compat.v1.lite"))

(defn RepresentativeDataset 
  "Representative dataset to evaluate optimizations.

  A representative dataset that can be used to evaluate optimizations by the
  converter. E.g. converter can use these examples to estimate (min, max) ranges
  by calibrating the model on inputs. This can allow converter to quantize a
  converted floating point model.
  "
  [ input_gen ]
  (py/call-attr lite "RepresentativeDataset"  input_gen ))
