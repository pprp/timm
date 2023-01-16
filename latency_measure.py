import torch 
import numpy as np 
# from timm.models.mobilenetv2 import mobilenet_v2
from ptflops import get_model_complexity_info
from torchvision.models import mobilenet_v2


model = mobilenet_v2()
device = torch.device("cuda")
model.to(device)
dummy_input = torch.randn(1, 3,224,224, dtype=torch.float).to(device)

# INIT LOGGERS
starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
repetitions = 300
timings=np.zeros((repetitions,1))

#GPU-WARM-UP
for _ in range(10):
    _ = model(dummy_input)

# MEASURE PERFORMANCE
with torch.no_grad():
    for rep in range(repetitions):
        starter.record()
        _ = model(dummy_input)
        ender.record()
        # WAIT FOR GPU SYNC
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender)
        timings[rep] = curr_time

mean_syn = np.sum(timings) / repetitions
std_syn = np.std(timings)
print("latency:", mean_syn, "+-", std_syn)

flops_count, param_count = get_model_complexity_info(
            model, (3, 224, 224), print_per_layer_stat=False)

print(f"Flops: {flops_count} \t Param: {param_count}")