# Fall_Detection
Code-base for fall-detection

read_raw_bin.py

Reads binary file generated by DCA1000EVM and processes them into a gif of range-doppler plots over time

### To generate npy files
1. Activate python virtual environment
```source ./venv/bin/activate```
2. Run wrapper that generates npy files for all bin files in folder
```python3 ./program_wrapper.py```
