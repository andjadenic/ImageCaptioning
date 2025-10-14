from skimage import io
from matplotlib import pyplot as plt
import torch


def display_image(img_path):
    img = io.imread(img_path)
    plt.figure(figsize=(5, 5))
    plt.axis('off')
    plt.imshow(img)
    plt.show()



def zero_after(x: torch.Tensor, id: int) -> torch.Tensor:
    mask_id = (x == id)

    # Find the first index of id in each row (or m if not found)
    idx_id = torch.argmax(mask_id.int(), dim=1)
    has_id = mask_id.any(dim=1)

    # Create a mask for zeroing
    n, m = x.shape
    arange = torch.arange(m, device=x.device).expand(n, m)
    idx_id_expanded = idx_id.unsqueeze(1).expand_as(x)

    # Build the final mask
    zero_mask = (arange >= idx_id_expanded) & has_id.unsqueeze(1)

    # Zero out elements
    x = x.clone()  # avoid modifying input in-place
    x[zero_mask] = 0
    return x
