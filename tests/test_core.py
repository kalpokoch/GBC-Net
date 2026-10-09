"""Fast CPU checks: model shapes, losses, metrics, config overrides, grouping."""

import numpy as np
import pandas as pd
import pytest
import torch

from gbcnet.config import apply_overrides
from gbcnet.losses import FocalLoss, MaskedAsymmetricLossMultiLabel
from gbcnet.metrics import DECISION_THRESHOLD, binary_metrics, multilabel_metrics
from gbcnet.models.registry import MODEL_REGISTRY, build_model, get_spec
from gbcnet.splits import build_group_ids


@pytest.mark.parametrize("name", sorted(MODEL_REGISTRY))
def test_model_forward_and_param_groups(name):
    model = build_model(name, pretrained=False).eval()
    with torch.no_grad():
        out = model(torch.randn(2, 1, 128, 128))
    assert out.shape == (2, 1)
    spec = get_spec(name)
    grouped = {id(p) for m in spec.backbone(model) + spec.attention(model) + spec.classifier(model) for p in m.parameters()}
    assert grouped == {id(p) for p in model.parameters()}, "optimizer groups must cover every parameter exactly"


def test_proposed_checkpoint_keys_match_notebook_layout():
    keys = build_model("ConvNeXtTiny_CBAM_MSAM", pretrained=False).state_dict().keys()
    assert "features.0.0.weight" in keys
    assert "cbam.channel_attention.fc.0.weight" in keys
    assert "msam.scale_convs.2.weight" in keys and "msam.fuse.bias" in keys
    assert "classifier.1.weight" in keys


def test_focal_loss_matches_bce_scale():
    logits = torch.tensor([[2.0], [-2.0]])
    targets = torch.tensor([[1.0], [0.0]])
    assert FocalLoss()(logits, targets).item() < torch.nn.functional.binary_cross_entropy_with_logits(logits, targets).item()


def test_masked_asl_ignores_masked_rows():
    loss = MaskedAsymmetricLossMultiLabel()
    logits = torch.randn(4, 3)
    targets = torch.randint(0, 2, (4, 3)).float()
    mask = torch.tensor([[1.0] * 3, [1.0] * 3, [0.0] * 3, [0.0] * 3])
    a = loss(logits, targets, mask)
    b = loss(logits[:2], targets[:2], mask[:2])
    assert torch.allclose(a, b)


def test_fixed_threshold_metrics():
    assert DECISION_THRESHOLD == 0.5
    t = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.6, 0.35, 0.9])
    m = binary_metrics(t, p)
    assert m["threshold"] == 0.5
    assert m["predictions"].tolist() == [0, 1, 0, 1]
    assert m["auc"] == pytest.approx(0.75)
    ml = multilabel_metrics(t[:, None].astype(float), p[:, None], ["x"], np.ones((4, 1)))
    assert ml.loc[0, "Threshold"] == 0.5 and ml.loc[0, "TP"] == 1


def test_overrides():
    cfg = apply_overrides({"train": {"epochs": 250}}, ["train.epochs=2", "seeds=[0]"])
    assert cfg["train"]["epochs"] == 2 and cfg["seeds"] == [0]


def test_body_mask_keeps_largest_region_and_fills_holes():
    from gbcnet.preprocessing import apply_body_mask, create_body_mask

    img = np.zeros((200, 200), dtype=np.uint8)
    img[40:160, 40:160] = 120   # body
    img[90:110, 90:110] = 5     # dark internal region (e.g. air) -> filled
    img[5:12, 5:12] = 200       # small bright artefact outside body -> removed
    mask = create_body_mask(img)
    assert mask[100, 100] == 255 and mask[8, 8] == 0
    out = apply_body_mask(img, mask)
    assert out[8, 8] == 0 and out[60, 60] == 120


def test_group_ids_merge_same_report_slices():
    span = '{"wall_thickening": ["asymmetric enhancing wall thickening of fundus"]}'
    df = pd.DataFrame({
        "relative_path": ["cancer/a/1.png", "cancer/a/2.png", "cancer/b/3.png", "cancer/a/4.png"],
        "matched_spans": [span, span, span, '{"x": ["short"]}'],
    })
    g = build_group_ids(df)
    assert g[0] == g[1] and g[0] != g[2] and g[3] not in (g[0], g[2])
