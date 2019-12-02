(ns tensorflow.-api.v1.compat.v1.losses.Reduction
  "Types of loss reduction.

  Contains the following values:

  * `NONE`: Un-reduced weighted losses with the same shape as input.
  * `SUM`: Scalar sum of weighted losses.
  * `MEAN`: Scalar `SUM` divided by sum of weights. DEPRECATED.
  * `SUM_OVER_BATCH_SIZE`: Scalar `SUM` divided by number of elements in losses.
  * `SUM_OVER_NONZERO_WEIGHTS`: Scalar `SUM` divided by number of non-zero
     weights. DEPRECATED.
  * `SUM_BY_NONZERO_WEIGHTS`: Same as `SUM_OVER_NONZERO_WEIGHTS`. DEPRECATED.
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce losses (import-module "tensorflow._api.v1.compat.v1.losses"))

(defn Reduction 
  "Types of loss reduction.

  Contains the following values:

  * `NONE`: Un-reduced weighted losses with the same shape as input.
  * `SUM`: Scalar sum of weighted losses.
  * `MEAN`: Scalar `SUM` divided by sum of weights. DEPRECATED.
  * `SUM_OVER_BATCH_SIZE`: Scalar `SUM` divided by number of elements in losses.
  * `SUM_OVER_NONZERO_WEIGHTS`: Scalar `SUM` divided by number of non-zero
     weights. DEPRECATED.
  * `SUM_BY_NONZERO_WEIGHTS`: Same as `SUM_OVER_NONZERO_WEIGHTS`. DEPRECATED.
  "
  [  ]
  (py/call-attr losses "Reduction"  ))
