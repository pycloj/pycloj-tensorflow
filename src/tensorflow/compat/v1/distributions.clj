(ns tensorflow.-api.v1.compat.v1.distributions
  "Core module for TensorFlow distribution objects and helpers.
"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce distributions (import-module "tensorflow._api.v1.compat.v1.distributions"))

(defn kl-divergence 
  "Get the KL-divergence KL(distribution_a || distribution_b). (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed after 2019-01-01.
Instructions for updating:
The TensorFlow Distributions library has moved to TensorFlow Probability (https://github.com/tensorflow/probability). You should update all references to use `tfp.distributions` instead of `tf.distributions`.

If there is no KL method registered specifically for `type(distribution_a)`
and `type(distribution_b)`, then the class hierarchies of these types are
searched.

If one KL method is registered between any pairs of classes in these two
parent hierarchies, it is used.

If more than one such registered method exists, the method whose registered
classes have the shortest sum MRO paths to the input types is used.

If more than one such shortest path exists, the first method
identified in the search is used (favoring a shorter MRO distance to
`type(distribution_a)`).

Args:
  distribution_a: The first distribution.
  distribution_b: The second distribution.
  allow_nan_stats: Python `bool`, default `True`. When `True`,
    statistics (e.g., mean, mode, variance) use the value \"`NaN`\" to
    indicate the result is undefined. When `False`, an exception is raised
    if one or more of the statistic's batch members are undefined.
  name: Python `str` name prefixed to Ops created by this class.

Returns:
  A Tensor with the batchwise KL-divergence between `distribution_a`
  and `distribution_b`.

Raises:
  NotImplementedError: If no KL method is defined for distribution types
    of `distribution_a` and `distribution_b`."
  [distribution_a distribution_b & {:keys [allow_nan_stats name]
                       :or {name None}} ]
    (py/call-attr-kw distributions "kl_divergence" [distribution_a distribution_b] {:allow_nan_stats allow_nan_stats :name name }))
