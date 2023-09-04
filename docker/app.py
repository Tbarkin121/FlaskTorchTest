from flask import Flask, jsonify, request
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import time


class PlanarArm(nn.Module):
    def __init__(self, num_segments):
        print('Init Arm')
        super(PlanarArm, self).__init__()
    
        self.num_segments = num_segments
        self.joint_angles = torch.ones(num_segments, requires_grad=True)
        self.joint_lengths =  torch.ones(num_segments, requires_grad=False)/num_segments
        self.xs = torch.zeros(num_segments+1, requires_grad=False)
        self.ys = torch.zeros(num_segments+1, requires_grad=False)
        self.x_targ=torch.tensor(-0.33, requires_grad=False)
        self.y_targ=torch.tensor(0.44, requires_grad=False)
        
        self.weights = torch.zeros([num_segments,1])
        self.weights[0] = 0
        

    @torch.jit.export
    def forward_kinematics(self):
        self.xs = torch.zeros(self.num_segments+1, requires_grad=False)
        self.ys = torch.zeros(self.num_segments+1, requires_grad=False)
        for s in range(1, self.num_segments+1):
            self.xs[s] = self.xs[s-1] + self.joint_lengths[s-1]*torch.cos(torch.sum(self.joint_angles[0:s]))
            self.ys[s] = self.ys[s-1] + self.joint_lengths[s-1]*torch.sin(torch.sum(self.joint_angles[0:s]))
                
    @torch.jit.export
    def get_residual(self):
        self.dx = self.xs[-1] - self.x_targ
        self.dy = self.ys[-1] - self.y_targ
        # error = torch.sqrt(dx**2 + dy**2)
    
    @torch.jit.export
    def compute_jacobian(self):
        # Compute forward kinematics
        self.forward_kinematics()
        self.get_residual()


        if self.joint_angles.grad is not None:
            self.joint_angles.grad = None

        self.dx.backward()
        self.jacobian_x = self.joint_angles.grad.clone()
        

        # Zero out the gradients before computing the next one        
        # self.joint_angles.grad = None
        if self.joint_angles.grad is not None:
            self.joint_angles.grad = None

        self.dy.backward()
        self.jacobian_y = self.joint_angles.grad.clone()
        
        self.J = torch.stack((env.jacobian_x, env.jacobian_y))

    @torch.jit.export
    def update_angles(self, dtheta):
        with torch.no_grad():
            self.joint_angles -= dtheta

    @torch.jit.export
    def control(self):
        m = 2
        n = self.num_segments
        gamma = .5
        
        self.compute_jacobian()
        
        JJT = torch.matmul(self.J, self.J.permute([1,0]))
        Im = torch.eye(m)
        R = torch.stack((env.dx, env.dy)).view(-1,1)
        M1 = torch.linalg.solve(JJT, self.J)
        M2 = torch.linalg.solve(JJT+gamma**2*Im, R)
        In = torch.eye(n)
        Zp = In - torch.matmul(self.J.permute([1,0]), M1)
        DeltaThetaPrimary = torch.matmul(self.J.permute([1,0]), M2)
        DeltaThetaSecondary = torch.matmul(Zp, self.joint_angles.view(-1,1) * self.weights)
        DeltaTheta = DeltaThetaPrimary + DeltaThetaSecondary        
        self.update_angles(DeltaTheta)
        
        
    @torch.jit.export
    def target_update(self, target):
        self.x_targ = target[0]
        self.y_targ = target[1]
        
    @torch.jit.export
    def get_angles(self):
        return self.joint_angles
    
    @torch.jit.export
    def get_positions(self):
        return torch.cat( (self.xs.view(-1,1), self.ys.view(-1,1)), dim=1)
    
# Global instance of PlanarArm
env = PlanarArm(3)
# Global instance of Flask app
application  = Flask(__name__)

@application.route('/')
def index():
    return "Hello, World!"

@application.route('/init', methods=['POST'])
def init():
    data = request.json
    n = data.get('n')
    
    global env
    env = PlanarArm(n)
    
    return f"Initialized with {n} segments!"

    
@application.route('/process_tensor', methods=['POST'])
def process_tensor():
    global env
    # Parse JSON data
    data = request.json
    input_list = data['tensor']
    
    # Convert list to PyTorch tensor
    tensor = torch.tensor(input_list)
    
    # Do some operations with the tensor
    # For this example, we'll just add 1 to the tensor
    result_tensor = tensor + 1
    
    # # Convert tensor to list
    # result_list = result_tensor.tolist()

    env.forward_kinematics()
    env.compute_jacobian()
    res = env.J

    # Convert tensor to list
    result_list = res.tolist()

    return jsonify({"result": result_list})


if __name__ == '__main__':
    application.run(host='0.0.0.0', port=5000, debug=True)