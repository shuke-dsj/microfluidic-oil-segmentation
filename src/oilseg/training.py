"""Configurable training with persisted splits, metrics and best checkpoint."""
import csv
import json
import math
import random
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from .data import OilDataset, paired_files
from .inference import pad_to_multiple, crop_to_original
from .metrics import confusion_matrix, class_iou
from .model import AttentionResUNet


class DiceCELoss(nn.Module):
    def forward(self, logits, target):
        ce = F.cross_entropy(logits, target)
        probabilities = logits.softmax(1)
        dice = []
        for index in range(logits.shape[1]):
            pred, truth = probabilities[:,index], (target==index).float()
            score = (2*(pred*truth).sum((1,2))+1e-6)/(pred.sum((1,2))+truth.sum((1,2))+1e-6)
            dice.append(1-score.mean())
        return .5*ce + .5*torch.stack(dice).mean()


def split_pairs(pairs, seed, split_file=None):
    if split_file:
        split = json.loads(Path(split_file).read_text(encoding='utf-8'))
        names = {p.name for p,_ in pairs}
        train,val = split['train'],split['val']
        if not train or not val or set(train)&set(val) or set(train)|set(val)!=names or len(train)+len(val)!=len(names):
            raise ValueError('Split must assign each image exactly once to non-empty train or val')
        lookup = {p.name:(p,m) for p,m in pairs}
        return [lookup[n] for n in train],[lookup[n] for n in val]
    # Same permutation convention as sklearn train_test_split(test_size=.2, random_state=seed).
    permutation = np.random.RandomState(seed).permutation(len(pairs))
    count = math.ceil(.2*len(pairs))
    if count == len(pairs):
        raise ValueError('At least two image pairs are required')
    return [pairs[i] for i in permutation[count:]],[pairs[i] for i in permutation[:count]]


def train(args, device):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type=='cuda':
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    train_pairs,val_pairs=split_pairs(paired_files(args.images,args.masks),args.seed,args.split)
    output=Path(args.output)
    output.mkdir(parents=True,exist_ok=True)
    (output/'split.json').write_text(json.dumps({'train':[p.name for p,_ in train_pairs], 'val':[p.name for p,_ in val_pairs]},indent=2),encoding='utf-8')
    config={k:v for k,v in vars(args).items()}
    config.update(torch_version=torch.__version__,device=str(device),train_count=len(train_pairs),val_count=len(val_pairs))
    (output/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    loader=DataLoader(OilDataset(train_pairs,True,args.crop_size),batch_size=args.batch_size,shuffle=True,num_workers=0)
    val_loader=DataLoader(OilDataset(val_pairs),batch_size=1,num_workers=0)
    model=AttentionResUNet(pretrained=not args.no_pretrained).to(device)
    loss_fn=DiceCELoss()
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.epochs,eta_min=1e-6)
    scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda')
    best=-1
    with (output/'training_metrics.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f)
        writer.writerow(['Epoch','Train_Loss','Val_mIoU'])
        for epoch in range(args.epochs):
            model.train()
            total=0
            for image,target in loader:
                image,target=image.to(device),target.to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device.type,enabled=device.type=='cuda'):
                    loss=loss_fn(model(image),target)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                total+=loss.item()
            scheduler.step()
            model.eval()
            values=[]
            with torch.inference_mode():
                for image,target in val_loader:
                    image,ph,pw=pad_to_multiple(image.to(device))
                    with torch.autocast(device.type,enabled=device.type=='cuda'):
                        pred=model(image).argmax(1)[0]
                    pred=crop_to_original(pred,ph,pw).cpu().numpy()
                    values.append(float(np.nanmean(class_iou(confusion_matrix(pred,target[0].numpy())))))
            value=float(np.mean(values))
            writer.writerow([epoch+1,total/len(loader),value])
            f.flush()
            if value>best:
                best=value
                torch.save(model.state_dict(),output/'best_attention_resunet.pth')
            print(f'epoch={epoch+1}/{args.epochs} loss={total/len(loader):.5f} val_image_mean_miou={value:.6f}',flush=True)
