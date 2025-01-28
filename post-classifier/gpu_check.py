# Testing if PyTorch is using GPU
import torch
print(f'{torch.cuda.current_device()} {torch.cuda.device_count()} {torch.cuda.get_device_name(0)}')