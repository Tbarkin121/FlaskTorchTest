import requests
import torch

# Create a tensor
tensor = torch.tensor([1.0, 2.0, 3.0, 4.0])

# Convert tensor to list and wrap in a JSON object
data = {"tensor": tensor.tolist()}

# Make a request to the Flask server
response = requests.post("http://127.0.0.1:5000/process_tensor", json=data)
# response = requests.post("http://pythontestenv.eba-258e7hvq.us-east-1.elasticbeanstalk.com//process_tensor", json=data)

# Convert the result back to a tensor
result_tensor = torch.tensor(response.json()['result'])
print(result_tensor)  # Should print tensor([2., 3., 4., 5.])