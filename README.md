### Fall detection
1. data_conversion  
   - Converts data between .bin, .gif, and .npy
2. ml
   - Contains functions that supports or enables ML functionalities
   - filters contains functions that
     - are applied before train and test sets are generated
     - maintains the shape of the input data and
     - maintains the meanings of each dimension (i.e. 3rd dimension represents time, 1st dimension represent doppler etc.)
   - feature_extract contains functions that
     - are applied before train and test sets are generated
     - are not filters
   - misc contains functions that are neither filters


### Logs
v3.0
- Add global and local normalization of range_profile() and doppler_profile()
- New filter - Downsampling of doppler axis by factor of 2
- Added ml_training - having only the filters/feature_extraction that is required to train SVM for final implementation, and most importantly transposes each frame before training and testing, and saves trained model weights

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
- Write functions to retrieve energy and range features (given by xavier)
