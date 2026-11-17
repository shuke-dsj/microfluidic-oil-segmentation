from pathlib import Path
import numpy as np
import pytest
import torch
from PIL import Image
from oilseg.labels import COLORS, rgb_to_labels
from oilseg.inference import pad_to_multiple, crop_to_original, export_prediction
from oilseg.metrics import confusion_matrix, class_iou
from oilseg.data import paired_files
from oilseg.model import AttentionResUNet
from oilseg.training import DiceCELoss, split_pairs


def test_palette_and_unknown_color():
    np.testing.assert_array_equal(rgb_to_labels(COLORS[None,:,:]),np.arange(6)[None,:])
    with pytest.raises(ValueError,match='palette'):
        rgb_to_labels(np.zeros((1,1,3)))


@pytest.mark.parametrize('shape',[(1,1),(17,9),(33,65),(32,64)])
def test_padding_restores_pixels(shape):
    image=torch.randn(1,3,*shape)
    padded,ph,pw=pad_to_multiple(image)
    assert padded.shape[-1]%32==padded.shape[-2]%32==0
    assert torch.equal(crop_to_original(padded,ph,pw),image)


def test_metric_aggregation_and_absent_classes():
    target=np.array([[0,0],[1,1]])
    pred=np.array([[0,1],[1,1]])
    matrix=confusion_matrix(pred,target)
    assert matrix[0,1]==1
    np.testing.assert_allclose(class_iou(matrix)[:2],[.5,2/3])
    assert np.isnan(class_iou(matrix)[2:]).all()


def test_pairing_rejects_wrong_names(tmp_path):
    for folder,name in [('images','a'),('masks','b')]:
        (tmp_path/folder).mkdir()
        Image.new('RGB',(2,2)).save(tmp_path/folder/f'{name}.png')
    with pytest.raises(ValueError,match='stems'):
        paired_files(tmp_path/'images',tmp_path/'masks')


def test_network_forward_and_training_gradients():
    torch.set_num_threads(2)
    model=AttentionResUNet(pretrained=False).eval()
    image=torch.rand(2,3,64,96)
    logits=model(image)
    assert logits.shape==(2,6,64,96)
    loss=DiceCELoss()(logits,torch.zeros(2,64,96,dtype=torch.long))
    loss.backward()
    assert model.att4.psi[0].weight.grad is not None
    assert torch.isfinite(loss)


def test_exports_keep_index_mask_and_normalize_oil(tmp_path):
    import csv
    pred=np.array([[0,2],[3,3]],dtype=np.uint8)
    export_prediction(pred,tmp_path,'demo')
    np.testing.assert_array_equal(np.array(Image.open(tmp_path/'demo_mask.png')),pred)
    with (tmp_path/'demo_areas.csv').open(encoding='utf-8') as f:
        rows=list(csv.DictReader(f))
    assert float(rows[3]['oil_fraction'])==pytest.approx(2/3)


def test_split_reproducible_and_nonoverlapping():
    pairs=[(Path(f'{i}.png'),Path(f'{i}.png')) for i in range(327)]
    train,val=split_pairs(pairs,42)
    assert (len(train),len(val))==(261,66)
    assert not set(train)&set(val)
    assert (train,val)==split_pairs(pairs,42)
