(ns tensorflow.python.keras.api.-v1.keras.initializers
  "Keras initializer serialization / deserialization.

"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce initializers (import-module "tensorflow.python.keras.api._v1.keras.initializers"))

(defn deserialize 
  "Return an `Initializer` object from its config."
  [ config custom_objects ]
  (py/call-attr initializers "deserialize"  config custom_objects ))

(defn get 
  ""
  [ identifier ]
  (py/call-attr initializers "get"  identifier ))

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

(defn serialize 
  ""
  [ initializer ]
  (py/call-attr initializers "serialize"  initializer ))
