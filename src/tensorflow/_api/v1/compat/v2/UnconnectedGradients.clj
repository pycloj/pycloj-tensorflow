(ns tensorflow.-api.v1.compat.v2.UnconnectedGradients
  "Controls how gradient computation behaves when y does not depend on x.

  The gradient of y with respect to x can be zero in two different ways: there
  could be no differentiable path in the graph connecting x to y (and so we can
  statically prove that the gradient is zero) or it could be that runtime values
  of tensors in a particular execution lead to a gradient of zero (say, if a
  relu unit happens to not be activated). To allow you to distinguish between
  these two cases you can choose what value gets returned for the gradient when
  there is no path in the graph from x to y:

  * `NONE`: Indicates that [None] will be returned if there is no path from x
    to y
  * `ZERO`: Indicates that a zero tensor will be returned in the shape of x.
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce v2 (import-module "tensorflow._api.v1.compat.v2"))
(defn UnconnectedGradients 
  "Controls how gradient computation behaves when y does not depend on x.

  The gradient of y with respect to x can be zero in two different ways: there
  could be no differentiable path in the graph connecting x to y (and so we can
  statically prove that the gradient is zero) or it could be that runtime values
  of tensors in a particular execution lead to a gradient of zero (say, if a
  relu unit happens to not be activated). To allow you to distinguish between
  these two cases you can choose what value gets returned for the gradient when
  there is no path in the graph from x to y:

  * `NONE`: Indicates that [None] will be returned if there is no path from x
    to y
  * `ZERO`: Indicates that a zero tensor will be returned in the shape of x.
  "
  [value names module qualname type  & {:keys [start]} ]
    (py/call-attr-kw v2 "UnconnectedGradients" [value names module qualname type] {:start start }))
