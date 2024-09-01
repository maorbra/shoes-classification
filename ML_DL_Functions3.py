import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
def ID1():
    '''
        Personal ID of the first student.
    '''
    # Insert your ID here
    return 316119767

def ID2():
    '''
        Personal ID of the second student. Fill this only if you were allowed to submit in pairs, Otherwise leave it zeros.
    '''
    # Insert your ID here
    return 000000000

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        

        
        # Parameters
        self.n = 16
        self.stride = 2
        self.kernel_size = 5  
        self.padding = int((self.kernel_size - 1) / 2.)

        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.n, kernel_size=self.kernel_size, padding=self.padding)
        self.conv2 = nn.Conv2d(in_channels=self.n, out_channels=self.n*2, kernel_size=self.kernel_size, padding=self.padding, stride=self.stride)
        self.conv3 = nn.Conv2d(in_channels=self.n*2, out_channels=self.n*4, kernel_size=self.kernel_size, padding=self.padding, stride=self.stride)
        self.conv4 = nn.Conv2d(in_channels=self.n*4, out_channels=self.n*8, kernel_size=self.kernel_size, padding=self.padding, stride=self.stride)
    
        self.fc1 = nn.Linear(self.n*8*28*14, 100)  # Adjusted based on the output size from the last conv layer
        self.fc2 = nn.Linear(100, 2)

        # Dropout layers
        self.drop1 = nn.Dropout(0.05)
        self.drop2 = nn.Dropout(0.03)

        # Batch Normalization layers
        self.batch_norm1 = nn.BatchNorm2d(self.n)
        self.batch_norm2 = nn.BatchNorm2d(2 * self.n)
        self.batch_norm3 = nn.BatchNorm2d(4 * self.n)
        self.batch_norm4 = nn.BatchNorm2d(8 * self.n)
        self.batch_norm5 = nn.BatchNorm1d(100)


    def forward(self, inp):
        out = self.conv1(inp)
        out = self.batch_norm1(out)
        out = torch.nn.functional.relu(out)
        out = torch.nn.functional.max_pool2d(out, 2)

        out = self.conv2(out)
        out = self.batch_norm2(out)
        out = torch.nn.functional.relu(out)

        out = self.conv3(out)
        out = self.batch_norm3(out)
        out = torch.nn.functional.relu(out)

        out = self.conv4(out)
        out = self.batch_norm4(out)
        out = torch.nn.functional.relu(out)

        out = out.reshape(-1, 8 * self.n * 28 * 14)

        out = self.drop1(out)
        out = self.fc1(out)
        out = self.batch_norm5(out)
        out = torch.nn.functional.relu(out)
        out = self.drop2(out)
        out = self.fc2(out)

        return out

class CNNChannel(nn.Module):
    def __init__(self):# Do NOT change the signature of this function
        super(CNNChannel, self).__init__()
        # TODO: complete this method
        n = int(11)
        kernel_size = int(5)
        padding = int((kernel_size - 1) / 2)
        
        self.conv1 = nn.Conv2d(in_channels=6, out_channels=n, kernel_size=kernel_size, stride=2, padding=padding)
        self.conv2 = nn.Conv2d(in_channels=n, out_channels=2*n, kernel_size=kernel_size, padding=padding)
        self.conv3 = nn.Conv2d(in_channels=2*n, out_channels=4*n, kernel_size=kernel_size, stride=2, padding=padding)
        self.conv4 = nn.Conv2d(in_channels=4*n, out_channels=8*n, kernel_size=kernel_size, padding=padding)
        self.fc1 = nn.Linear(in_features=8*n*14*14, out_features=100)
        self.fc2 = nn.Linear(in_features=100, out_features=2)
        

    # TODO: complete this class
    def forward(self,inp):# Do NOT change the signature of this function
        '''
          prerequests:
          parameter inp: the input image, pytorch tensor
          inp.shape == (N,3,448,224):
            N   := batch size
            3   := RGB channels
            448 := Height
            224 := Width
          
          return output, pytorch tensor
          output.shape == (N,2):
            N := batch size
            2 := same/different pair
        '''
        # TODO start by changing the shape of the input to (N,6,224,224)
        # TODO: complete this function
        
        # Move all parametes tensors to the same device as the inp tensor
        for param in self.parameters():
            param.data = param.data.to(inp.device)
        
        n = int(11)
        inp = torch.cat((inp[:, :, :224, :], inp[:, :, 224:, :]), dim=1)

        out = self.conv1(inp)
        out = torch.nn.functional.relu(out)

        out = self.conv2(out)
        out = torch.nn.functional.relu(out)
        out = torch.nn.functional.max_pool2d(out, kernel_size=2)

        out = self.conv3(out)
        out = torch.nn.functional.relu(out)

        out = self.conv4(out)
        out = torch.nn.functional.relu(out)
        out = torch.nn.functional.max_pool2d(out, kernel_size=2)

        out = out.contiguous().view(-1, 8*n*14*14)
        out = self.fc1(out)
        out = torch.nn.functional.relu(out)
        
        out = self.fc2(out)
                
        return out
