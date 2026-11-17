"""Public command-line interface; no machine-specific paths."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from .data import image_files, paired_files
from .labels import CLASS_NAMES, read_mask
from .inference import load_model, predict, export_prediction
from .metrics import confusion_matrix, class_iou


def sha256(path):
    digest=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):
            digest.update(chunk)
    return digest.hexdigest()


def safe_values(values):
    return [float(x) if np.isfinite(x) else None for x in values]


def main():
    parser=argparse.ArgumentParser(description='Microfluidic residual-oil segmentation')
    commands=parser.add_subparsers(dest='command',required=True)
    for name in ('predict','evaluate','train'):
        p=commands.add_parser(name)
        p.add_argument('--images',required=True,help='Single image (predict only) or image directory')
        p.add_argument('--output',default=f'outputs/{name}')
        p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
        if name!='train':
            p.add_argument('--weights',required=True)
            p.add_argument('--amp',action='store_true',help='CUDA mixed precision (default FP32)')
        if name!='predict':
            p.add_argument('--masks',required=True)
        if name=='train':
            p.add_argument('--epochs',type=int,default=50)
            p.add_argument('--batch-size',type=int,default=8)
            p.add_argument('--crop-size',type=int,default=512)
            p.add_argument('--lr',type=float,default=1e-4)
            p.add_argument('--seed',type=int,default=42)
            p.add_argument('--split',help='JSON train/val filename lists; use grouped splits for new evaluations')
            p.add_argument('--no-pretrained',action='store_true')
    args=parser.parse_args()
    device=torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device)
    if args.command=='train':
        if min(args.epochs,args.batch_size,args.crop_size)<=0 or args.crop_size%32 or args.lr<=0:
            parser.error('Positive epochs, batch size and LR required; crop size must be a positive multiple of 32')
        from .training import train
        return train(args,device)
    model=load_model(args.weights,device)
    output=Path(args.output)
    output.mkdir(parents=True,exist_ok=True)
    if args.command=='predict':
        paths=[Path(args.images)] if Path(args.images).is_file() else image_files(args.images)
        for path in paths:
            with Image.open(path) as image:
                pred=predict(model,image,device,args.amp)
            export_prediction(pred,output,path.stem)
        print(f'Exported {len(paths)} predictions to {output}')
        return
    matrix=np.zeros((6,6),dtype=np.int64)
    rows=[]
    for image_path,mask_path in paired_files(args.images,args.masks):
        with Image.open(image_path) as image:
            pred=predict(model,image,device,args.amp)
        target=read_mask(mask_path)
        current=confusion_matrix(pred,target)
        matrix+=current
        iou=class_iou(current)
        rows.append({'image':image_path.name,'image_miou':float(np.nanmean(iou)),**{f'iou_{n}':v for n,v in zip(CLASS_NAMES,safe_values(iou))}})
    with (output/'per_image.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result={'images':len(rows),'image_mean_miou':float(np.mean([r['image_miou'] for r in rows])),
            'global_miou':float(np.nanmean(class_iou(matrix))),
            'pixel_accuracy':float(np.trace(matrix)/matrix.sum()),
            'class_iou':dict(zip(CLASS_NAMES,safe_values(class_iou(matrix)))),
            'class_pixels':dict(zip(CLASS_NAMES,[int(x) for x in matrix.sum(1)])),
            'confusion_matrix_target_rows':matrix.tolist(),'weights_sha256':sha256(args.weights),
            'torch_version':str(torch.__version__),'device':str(device),'amp':args.amp}
    (output/'evaluation.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':
    main()
