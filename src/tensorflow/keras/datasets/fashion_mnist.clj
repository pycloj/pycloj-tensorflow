(ns tensorflow.python.keras.api.-v1.keras.datasets.fashion-mnist
  "Fashion-MNIST dataset.

"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce fashion-mnist (import-module "tensorflow.python.keras.api._v1.keras.datasets.fashion_mnist"))

(defn load-data 
  "Loads the Fashion-MNIST dataset.

  Returns:
      Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.

  License:
      The copyright for Fashion-MNIST is held by Zalando SE.
      Fashion-MNIST is licensed under the [MIT license](
      https://github.com/zalandoresearch/fashion-mnist/blob/master/LICENSE).

  "
  [  ]
  (py/call-attr fashion-mnist "load_data"  ))
