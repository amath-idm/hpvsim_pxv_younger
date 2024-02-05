# hpvsim_txvx_analyses - Serin Version

Major updates
- Added worklog_latency model.pptx that sums up all previous slides and future recommendations
- Added/modified python files 
  - analyze_calibration.py
    - read calibration results from failed study that has .tmp files but not .obj file
    - improved plotting functions like plot_density, pairplotpars
    - parameter analysis - with fit_model and correlation between parameters
  - run_scenarios.py : save msims results as obj. Read obj file and select important outcomes from organize_msim_results
  - analyze_scenarios.py : plot diverse results by desired outcomes, with filtering and plot_by variables. as well as plot by age
