import numpy as np
class Conv2D():
    def __init__(self, kernel_size):
        self.kernel = np.random.randn(kernel_size[0], kernel_size[1])
    
    def forward(self, input):
        x_size = self.kernel.shape[0]
        y_size = self.kernel.shape[1]
        
        output_x = input.shape[0] - self.kernel.shape[0] + 1
        output_y = input.shape[1] - self.kernel.shape[1] + 1
        
        self.output_matrix = np.zeros((output_x, output_y))
        
        for i in range(output_x):
            for j in range(output_y):
                sub_matrix = input[i:i+x_size, j:j+y_size]
                self.output_matrix[i,j] = np.sum(sub_matrix * self.kernel)
                
        return self.output_matrix
    
    def backward(self, dvalues):
        x_size, y_size = self.kernel.shape
        
        output_x, output_y = dvalues.shape
        
        self.d_kernel = np.zeros_like(self.kernel)
        self.dinput = np.zeros_like(self.input)
        
        for i in range(output_x):
            for j in range(output_y):
                sub_matrix = self.input[i:i+x_size, j:j+y_size]
                
                # how much each kernel value contributes to the loss? (claude comment)
                self.d_kernel += sub_matrix * dvalues[i,j]
                
                # how much each input pixel contributes to the loss
                self.dinput[i:i+x_size, j:j+y_size] += self.kernel * dvalues[i,j]
        
        return self.dinput
        
