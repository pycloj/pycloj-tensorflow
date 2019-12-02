(ns tensorflow-core.contrib.stat-summarizer.StatSummarizer
  ""
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce stat-summarizer (import-module "tensorflow_core.contrib.stat_summarizer"))

(defn StatSummarizer 
  ""
  [  ]
  (py/call-attr stat-summarizer "StatSummarizer"  ))

(defn GetOutputString 
  ""
  [ self  ]
  (py/call-attr self "GetOutputString"  self  ))

(defn PrintStepStats 
  ""
  [ self  ]
  (py/call-attr self "PrintStepStats"  self  ))

(defn ProcessStepStats 
  ""
  [ self step_stats ]
  (py/call-attr self "ProcessStepStats"  self step_stats ))

(defn ProcessStepStatsStr 
  ""
  [ self step_stats_str ]
  (py/call-attr self "ProcessStepStatsStr"  self step_stats_str ))
