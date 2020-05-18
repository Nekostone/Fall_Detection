### Fall detection
1. data_conversion
   - Converts data between .bin, .gif, and .npy
2. ml
   - Contains functions that supports or enables ML functionalities

### Logs
v2.0
- Restructuring of source code
  - three phases: 
    - preprocessing: Output data must still have the same shape as input data
      - Each data point should have features and labels, and should look something like the format below:
        ```[
             [
               [DATA OF ONE FRAME],
               [DATA OF ONE FRAME]
             ], 
             [LABEL_NUMBER]]```
    - feature extraction: Output data shape can vary depending on model
    - model
  - each phases are called as imported functions instead of being executed as individual files
    - allow for shared variables in the overall wrapping function
    - simpler setup and teardown for testing of functions
  - labelling will be embedded into each datapoint (not depending on parent dir names)
- Make remove center 3 lines a filter in the preprocessing phase
- Write function to label data from csv file
- More tests

