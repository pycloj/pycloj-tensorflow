(ns tensorflow.contrib.distributions.RegisterKL
  "Decorator to register a KL divergence implementation function.

  Usage:

  @distributions.RegisterKL(distributions.Normal, distributions.Normal)
  def _kl_normal_mvn(norm_a, norm_b):
    # Return KL(norm_a || norm_b)
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

(defn RegisterKL 
  "Decorator to register a KL divergence implementation function.

  Usage:

  @distributions.RegisterKL(distributions.Normal, distributions.Normal)
  def _kl_normal_mvn(norm_a, norm_b):
    # Return KL(norm_a || norm_b)
  "
  [ dist_cls_a dist_cls_b ]
  (py/call-attr distributions "RegisterKL"  dist_cls_a dist_cls_b ))
