(ns tensorflow.contrib.distributions.ReparameterizationType
  "Instances of this class represent how sampling is reparameterized.

  Two static instances exist in the distributions library, signifying
  one of two possible properties for samples from a distribution:

  `FULLY_REPARAMETERIZED`: Samples from the distribution are fully
    reparameterized, and straight-through gradients are supported.

  `NOT_REPARAMETERIZED`: Samples from the distribution are not fully
    reparameterized, and straight-through gradients are either partially
    unsupported or are not supported at all. In this case, for purposes of
    e.g. RL or variational inference, it is generally safest to wrap the
    sample results in a `stop_gradients` call and use policy
    gradients / surrogate loss instead.
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce distributions (import-module "tensorflow.contrib.distributions"))

(defn ReparameterizationType 
  "Instances of this class represent how sampling is reparameterized.

  Two static instances exist in the distributions library, signifying
  one of two possible properties for samples from a distribution:

  `FULLY_REPARAMETERIZED`: Samples from the distribution are fully
    reparameterized, and straight-through gradients are supported.

  `NOT_REPARAMETERIZED`: Samples from the distribution are not fully
    reparameterized, and straight-through gradients are either partially
    unsupported or are not supported at all. In this case, for purposes of
    e.g. RL or variational inference, it is generally safest to wrap the
    sample results in a `stop_gradients` call and use policy
    gradients / surrogate loss instead.
  "
  [ rep_type ]
  (py/call-attr distributions "ReparameterizationType"  rep_type ))
