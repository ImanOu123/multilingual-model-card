import requests

response = requests.post("http://localhost:8765/translate", json={"text": "1: For all the methods, we used <mark>10-fold cross validation</mark> (i.e., each fold we have 556 training and 62 test samples) to tune free parameters, e.g., the kernel form and parameters for GPOR and LapSVM. Note that all the alternative methods stack X and Z together into a whole data matrix and ignore their heterogeneous nature.<br>2: Features associated one-to-one with a vertical (Clarity, ReDDE, the query likelihood given the vertical's query-log and Soft.ReDDE) were normalized across verticals before scaling. Supervised training/testing was done via <mark>10-fold cross validation</mark>. Parameter τ was tuned for each training fold on the same 500 query validation set used for our single feature baselines.<br>"})
print(response.json())


# curl -X POST "http://localhost:8765/translate" -H "Content-Type: application/json" -d '{"text": "Hello, how are you?"}'
