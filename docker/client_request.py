import requests
import torch
import time


# url = "http://10.0.0.231:5000"
url = "http://pythontest-env-2.eba-5ibfyhpw.us-east-1.elasticbeanstalk.com/"

t_start = time.perf_counter()
# Make a request to the Flask server
data = {"n": 5}
response = requests.post(url+"/init", json=data)

# Create a tensor
tensor = torch.tensor([1.0, 2.0, 3.0, 4.0])
# Convert tensor to list and wrap in a JSON object
data = {"tensor": tensor.tolist()}
response = requests.post(url+"/process_tensor", json=data)
# response = requests.post("http://pythontestenv.eba-258e7hvq.us-east-1.elasticbeanstalk.com//process_tensor", json=data)

# Convert the result back to a tensor
result_tensor = torch.tensor(response.json()['result'])
t_end = time.perf_counter()
t_diff = t_end-t_start
print(result_tensor)  # Should print tensor([2., 3., 4., 5.])
print(f'Elapsed Time = {t_diff}')