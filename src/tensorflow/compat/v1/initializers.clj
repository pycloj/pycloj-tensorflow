(ns tensorflow.-api.v1.compat.v1.initializers
  "Public API for tf.initializers namespace.
"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce initializers (import-module "tensorflow._api.v1.compat.v1.initializers"))

(defn global-variables 
  "Returns an Op that initializes global variables.

  This is just a shortcut for `variables_initializer(global_variables())`

  Returns:
    An Op that initializes global variables in the graph.
  "
  [  ]
  (py/call-attr initializers "global_variables"  ))

(defn he-normal 
  "He normal initializer.

  It draws samples from a truncated normal distribution centered on 0
  with standard deviation (after truncation) given by
  `stddev = sqrt(2 / fan_in)` where `fan_in` is the number of
  input units in the weight tensor.

  Arguments:
      seed: A Python integer. Used to seed the random generator.

  Returns:
      An initializer.

  References:
      [He et al., 2015]
      (https://www.cv-foundation.org/openaccess/content_iccv_2015/html/He_Delving_Deep_into_ICCV_2015_paper.html)
      # pylint: disable=line-too-long
      ([pdf](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/He_Delving_Deep_into_ICCV_2015_paper.pdf))
  "
  [ seed ]
  (py/call-attr initializers "he_normal"  seed ))

(defn he-uniform 
  "He uniform variance scaling initializer.

  It draws samples from a uniform distribution within [-limit, limit]
  where `limit` is `sqrt(6 / fan_in)`
  where `fan_in` is the number of input units in the weight tensor.

  Arguments:
      seed: A Python integer. Used to seed the random generator.

  Returns:
      An initializer.

  References:
      [He et al., 2015]
      (https://www.cv-foundation.org/openaccess/content_iccv_2015/html/He_Delving_Deep_into_ICCV_2015_paper.html)
      # pylint: disable=line-too-long
      ([pdf](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/He_Delving_Deep_into_ICCV_2015_paper.pdf))
  "
  [ seed ]
  (py/call-attr initializers "he_uniform"  seed ))

(defn lecun-normal 
  "LeCun normal initializer.

  It draws samples from a truncated normal distribution centered on 0
  with standard deviation (after truncation) given by
  `stddev = sqrt(1 / fan_in)` where `fan_in` is the number of
  input units in the weight tensor.

  Arguments:
      seed: A Python integer. Used to seed the random generator.

  Returns:
      An initializer.

  References:
      - Self-Normalizing Neural Networks,
      [Klambauer et al.,
      2017](https://papers.nips.cc/paper/6698-self-normalizing-neural-networks)
      # pylint: disable=line-too-long
      ([pdf](https://papers.nips.cc/paper/6698-self-normalizing-neural-networks.pdf))
      - Efficient Backprop,
      [Lecun et al., 1998](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf)
  "
  [ seed ]
  (py/call-attr initializers "lecun_normal"  seed ))

(defn lecun-uniform 
  "LeCun uniform initializer.

  It draws samples from a uniform distribution within [-limit, limit]
  where `limit` is `sqrt(3 / fan_in)`
  where `fan_in` is the number of input units in the weight tensor.

  Arguments:
      seed: A Python integer. Used to seed the random generator.

  Returns:
      An initializer.

  References:
      - Self-Normalizing Neural Networks,
      [Klambauer et al.,
      2017](https://papers.nips.cc/paper/6698-self-normalizing-neural-networks)
      # pylint: disable=line-too-long
      ([pdf](https://papers.nips.cc/paper/6698-self-normalizing-neural-networks.pdf))
      - Efficient Backprop,
      [Lecun et al., 1998](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf)
  "
  [ seed ]
  (py/call-attr initializers "lecun_uniform"  seed ))

(defn local-variables 
  "Returns an Op that initializes all local variables.

  This is just a shortcut for `variables_initializer(local_variables())`

  Returns:
    An Op that initializes all local variables in the graph.
  "
  [  ]
  (py/call-attr initializers "local_variables"  ))

(defn tables-initializer 
  "Returns an Op that initializes all tables of the default graph.

  See the [Low Level
  Intro](https://www.tensorflow.org/guide/low_level_intro#feature_columns)
  guide, for an example of usage.

  Args:
    name: Optional name for the initialization op.

  Returns:
    An Op that initializes all tables.  Note that if there are
    not tables the returned Op is a NoOp.
  "
  [ & {:keys [name]} ]
   (py/call-attr-kw initializers "tables_initializer" [] {:name name }))
(defn variables 
  "Returns an Op that initializes a list of variables.

  After you launch the graph in a session, you can run the returned Op to
  initialize all the variables in `var_list`. This Op runs all the
  initializers of the variables in `var_list` in parallel.

  Calling `initialize_variables()` is equivalent to passing the list of
  initializers to `Group()`.

  If `var_list` is empty, however, the function still returns an Op that can
  be run. That Op just has no effect.

  Args:
    var_list: List of `Variable` objects to initialize.
    name: Optional name for the returned operation.

  Returns:
    An Op that run the initializers of all the specified variables.
  "
  [var_list  & {:keys [name]} ]
    (py/call-attr-kw initializers "variables" [var_list] {:name name }))
