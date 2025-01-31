# Load the Model and Generate Images
This script loads the pre trained model, performs inference and visualizes the generated images for a set of predefined prompts.

### To get started, install the necessary dependencies:
```python
!pip install diffusers peft huggingface_hub torch torchvision matplotlib
```

```python
import matplotlib.pyplot as plt
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
from peft import PeftModel
```

### Function to visualize generated images
```python
def visualize_images(images):
    assert len(images) == 4
    plt.figure(figsize=(12, 3))
    for i, image in enumerate(images):
        plt.subplot(1, 4, i+1)
        plt.imshow(image)
        plt.axis('off')
    plt.subplots_adjust(wspace=-0.01, hspace=-0.01)
```
### Load the model
```python
device = "cpu"  # gpu, ...
```
```python
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float32  # Load model in FP32
)
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
pipe.scheduler.timesteps = pipe.scheduler.timesteps
pipe.scheduler.alphas_cumprod = pipe.scheduler.alphas_cumprod

pipe.unet = PeftModel.from_pretrained(
    pipe.unet,
    "evetrubetskaya/consistency_distillation",
    subfolder='multi-cd',
    adapter_name="multi-cd",
)
pipe = pipe.to(device)
pipe.unet.eval()
```
### Set inference parameters

```python
guidance_scale = 1

validation_prompts = [
    "A sad puppy with large eyes",
    ...
]
```

### Generate and visualize images
```python
for prompt in validation_prompts:
    images = pipe(
        prompt=prompt,
        num_inference_steps=4,
        guidance_scale=guidance_scale,
        num_images_per_prompt=4
    ).images
    
    visualize_images(images)
```

*Hope you have generated cute puppies and other things* 🐶