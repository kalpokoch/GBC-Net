"""Print parameter counts for every binary registry model.

    python -m gbcnet.count_params
"""

from __future__ import annotations

from .models.registry import MODEL_REGISTRY, build_model


def main():
    print(f"{'Model':<26} {'Total':>14} {'Backbone':>14} {'Attention':>12} {'Head':>10}")
    print("-" * 80)
    for name, spec in MODEL_REGISTRY.items():
        model = build_model(name, pretrained=False)

        def count(modules):
            return sum(p.numel() for m in modules for p in m.parameters())

        total = sum(p.numel() for p in model.parameters())
        print(f"{name:<26} {total:>14,} {count(spec.backbone(model)):>14,} "
              f"{count(spec.attention(model)):>12,} {count(spec.classifier(model)):>10,}")


if __name__ == "__main__":
    main()
