"""Full-resolution inference, safe checkpoint loading, and quantitative exports."""
from pathlib import Path
import csv
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from .labels import COLORS, CLASS_NAMES


def pad_to_multiple(x, multiple=32):
    if multiple <= 0 or min(x.shape[-2:]) < 1:
        raise ValueError('Positive multiple and non-empty spatial dimensions required')
    h, w = x.shape[-2:]
    ph, pw = (-h) % multiple, (-w) % multiple
    # Reflect padding requires each padding width to be smaller than its dimension.
    mode = 'reflect' if ph < h and pw < w else 'replicate'
    return F.pad(x, (0, pw, 0, ph), mode=mode) if ph or pw else x, ph, pw


def crop_to_original(x, pad_h, pad_w):
    h, w = x.shape[-2:]
    return x[..., :h-pad_h, :w-pad_w]


def load_model(weights, device):
    from .model import AttentionResUNet
    model = AttentionResUNet(pretrained=False).to(device)
    model.load_state_dict(torch.load(weights, map_location=device, weights_only=True), strict=True)
    return model.eval()


@torch.inference_mode()
def predict(model, image, device, amp=False):
    image = image.convert('RGB')
    tensor = TF.to_tensor(image).unsqueeze(0).to(device)
    tensor, ph, pw = pad_to_multiple(tensor)
    with torch.autocast(device.type, enabled=amp and device.type == 'cuda'):
        logits = model(tensor)
    return crop_to_original(logits.argmax(1)[0], ph, pw).cpu().numpy().astype(np.uint8)


def export_prediction(prediction, output, stem):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    Image.fromarray(prediction).save(output / f'{stem}_mask.png')
    Image.fromarray(COLORS[prediction]).save(output / f'{stem}_color.png')
    oil_pixels = np.count_nonzero(prediction >= 2)
    rows = []
    for index, name in enumerate(CLASS_NAMES):
        count = int(np.count_nonzero(prediction == index))
        rows.append(dict(class_id=index, class_name=name, pixels=count,
                         image_fraction=count / prediction.size,
                         oil_fraction=count / oil_pixels if index >= 2 and oil_pixels else ''))
    with (output / f'{stem}_areas.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
