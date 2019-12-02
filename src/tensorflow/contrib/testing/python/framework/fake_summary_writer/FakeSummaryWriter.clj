(ns tensorflow.contrib.testing.python.framework.fake-summary-writer.FakeSummaryWriter
  "Fake summary writer."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce fake-summary-writer (import-module "tensorflow.contrib.testing.python.framework.fake_summary_writer"))

(defn FakeSummaryWriter 
  "Fake summary writer."
  [ logdir graph ]
  (py/call-attr fake-summary-writer "FakeSummaryWriter"  logdir graph ))

(defn add-graph 
  "Add graph."
  [ self graph global_step graph_def ]
  (py/call-attr self "add_graph"  self graph global_step graph_def ))

(defn add-meta-graph 
  "Add metagraph."
  [ self meta_graph_def global_step ]
  (py/call-attr self "add_meta_graph"  self meta_graph_def global_step ))

(defn add-run-metadata 
  ""
  [ self run_metadata tag global_step ]
  (py/call-attr self "add_run_metadata"  self run_metadata tag global_step ))

(defn add-session-log 
  ""
  [ self session_log global_step ]
  (py/call-attr self "add_session_log"  self session_log global_step ))

(defn add-summary 
  "Add summary."
  [ self summ current_global_step ]
  (py/call-attr self "add_summary"  self summ current_global_step ))

(defn assert-summaries 
  "Assert expected items have been added to summary writer."
  [ self test_case expected_logdir expected_graph expected_summaries expected_added_graphs expected_added_meta_graphs expected_session_logs ]
  (py/call-attr self "assert_summaries"  self test_case expected_logdir expected_graph expected_summaries expected_added_graphs expected_added_meta_graphs expected_session_logs ))

(defn close 
  ""
  [ self  ]
  (py/call-attr self "close"  self  ))

(defn flush 
  ""
  [ self  ]
  (py/call-attr self "flush"  self  ))

(defn reopen 
  ""
  [ self  ]
  (py/call-attr self "reopen"  self  ))

(defn summaries 
  ""
  [ self ]
    (py/call-attr self "summaries"))
