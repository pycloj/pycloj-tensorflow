(ns tensorflow.python.keras.api.-v1.keras.utils.Progbar
  "Displays a progress bar.

  Arguments:
      target: Total number of steps expected, None if unknown.
      width: Progress bar width on screen.
      verbose: Verbosity mode, 0 (silent), 1 (verbose), 2 (semi-verbose)
      stateful_metrics: Iterable of string names of metrics that
          should *not* be averaged over time. Metrics in this list
          will be displayed as-is. All others will be averaged
          by the progbar before display.
      interval: Minimum visual progress update interval (in seconds).
      unit_name: Display name for step counts (usually \"step\" or \"sample\").
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce utils (import-module "tensorflow.python.keras.api._v1.keras.utils"))

(defn Progbar 
  "Displays a progress bar.

  Arguments:
      target: Total number of steps expected, None if unknown.
      width: Progress bar width on screen.
      verbose: Verbosity mode, 0 (silent), 1 (verbose), 2 (semi-verbose)
      stateful_metrics: Iterable of string names of metrics that
          should *not* be averaged over time. Metrics in this list
          will be displayed as-is. All others will be averaged
          by the progbar before display.
      interval: Minimum visual progress update interval (in seconds).
      unit_name: Display name for step counts (usually \"step\" or \"sample\").
  "
  [target & {:keys [width verbose interval stateful_metrics unit_name]
                       :or {stateful_metrics None}} ]
    (py/call-attr-kw utils "Progbar" [target] {:width width :verbose verbose :interval interval :stateful_metrics stateful_metrics :unit_name unit_name }))

(defn add 
  ""
  [ self n values ]
  (py/call-attr self "add"  self n values ))

(defn update 
  "Updates the progress bar.

    Arguments:
        current: Index of current step.
        values: List of tuples:
            `(name, value_for_last_step)`.
            If `name` is in `stateful_metrics`,
            `value_for_last_step` will be displayed as-is.
            Else, an average of the metric over time will be displayed.
    "
  [ self current values ]
  (py/call-attr self "update"  self current values ))
