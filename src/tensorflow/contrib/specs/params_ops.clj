(ns tensorflow.contrib.specs.python.params-ops
  "Operators for concise TensorFlow parameter specifications.

This module is used as an environment for evaluating expressions
in the \"params\" DSL.

Specifications are intended to assign simple numerical
values. Examples:

    --params \"n=64; d=5\" --spec \"(Cr(n) | Mp([2, 2])) ** d | Fm\"

The random parameter primitives are useful for running large numbers
of experiments with randomly distributed parameters:

    --params \"n=Li(5,500); d=Ui(1,5)\" --spec \"(Cr(n) | Mp([2, 2])) ** d | Fm\"

Internally, this might be implemented as follows:

    params = specs.create_params(FLAGS.params, {})
    logging.info(repr(params))
    net = specs.create_net(FLAGS.spec, inputs, params)

Note that separating the specifications into parameters and network
creation allows us to log the random parameter values easily.

The implementation of this will change soon in order to support
hyperparameter tuning with steering. Instead of returning a number,
the primitives below will return a class instance that is then
used to generate a random number by the framework.
"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce params-ops (import-module "tensorflow.contrib.specs.python.params_ops"))

(defn Lf 
  "Log-uniform distributed floatint point number."
  [ lo hi ]
  (py/call-attr params-ops "Lf"  lo hi ))

(defn Li 
  "Log-uniform distributed integer, inclusive limits."
  [ lo hi ]
  (py/call-attr params-ops "Li"  lo hi ))
(defn Nt 
  "Normally distributed floating point number with truncation."
  [mu sigma  & {:keys [limit]} ]
    (py/call-attr-kw params-ops "Nt" [mu sigma] {:limit limit }))

(defn Uf 
  "Uniformly distributed floating number."
  [ & {:keys [lo hi]} ]
   (py/call-attr-kw params-ops "Uf" [] {:lo lo :hi hi }))

(defn Ui 
  "Uniformly distributed integer, inclusive limits."
  [ lo hi ]
  (py/call-attr params-ops "Ui"  lo hi ))
