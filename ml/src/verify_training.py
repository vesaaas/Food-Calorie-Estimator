"""Short Phase 3 checks using real train/validation batches; never evaluate test data."""

import argparse
from contextlib import redirect_stdout
from dataclasses import asdict
import io
import json
from pathlib import Path
import random
import tempfile
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

if __package__:
    from . import checkpoint as checkpoint_module, model as model_module, train as train_module
    from .checkpoint import load_checkpoint, save_checkpoint
    from .dataset import DataConfig, DEVELOPMENT_CLASSES, PROCESSED_DIR, create_data_pipeline
    from .model import IMAGENET_WEIGHTS, ModelConfig, create_model
    from .runtime import seed_training, select_device
    from .training import EpochMetrics, TrainingConfig, train_one_epoch, validate
    from .verify_pipeline import require
else:
    import checkpoint as checkpoint_module
    import model as model_module
    import train as train_module
    from checkpoint import load_checkpoint, save_checkpoint
    from dataset import DataConfig, DEVELOPMENT_CLASSES, PROCESSED_DIR, create_data_pipeline
    from model import IMAGENET_WEIGHTS, ModelConfig, create_model
    from runtime import seed_training, select_device
    from training import EpochMetrics, TrainingConfig, train_one_epoch, validate
    from verify_pipeline import require


def expect_error(error_type, operation, message: str) -> None:
    try:
        operation()
    except error_type:
        return
    raise AssertionError(message)


def verify_runtime() -> None:
    for mps, cuda, expected in ((True, True, "mps"), (True, False, "mps"), (False, True, "cuda"), (False, False, "cpu")):
        with patch("torch.backends.mps.is_available", return_value=mps), patch("torch.cuda.is_available", return_value=cuda):
            require(select_device().type == expected, "Incorrect automatic device priority")
            require(select_device("cpu").type == "cpu", "Explicit CPU selection failed")
            for requested, available in (("mps", mps), ("cuda", cuda)):
                if available:
                    require(select_device(requested).type == requested, "Explicit device selection failed")
                else:
                    expect_error(RuntimeError, lambda: select_device(requested), "Unavailable device silently accepted")
    expect_error(ValueError, lambda: select_device("invalid"), "Invalid device accepted")
    seed_training()
    first = random.random(), np.random.rand(), torch.rand(3)
    seed_training()
    second = random.random(), np.random.rand(), torch.rand(3)
    require(first[:2] == second[:2] and torch.equal(first[2], second[2]), "Seed reproduction failed")
    print("PASS: device selection/override/error branches (mocked availability); Python/NumPy/PyTorch seed reproduction", flush=True)


def verify_pretrained_contract() -> None:
    """Check weight enum selection and failure reporting without network access."""
    with redirect_stdout(io.StringIO()), patch.object(model_module, "efficientnet_b0") as factory:
        factory.return_value = nn.Module()
        factory.return_value.classifier = nn.Sequential(nn.Dropout(), nn.Linear(1280, 1000))
        candidate = create_model(ModelConfig(pretrained=True))
        require(factory.call_args.kwargs["weights"] is IMAGENET_WEIGHTS, "Wrong pretrained weights")
        require(candidate.classifier[1].out_features == len(DEVELOPMENT_CLASSES), "Wrong pretrained head")
    with redirect_stdout(io.StringIO()), patch.object(model_module, "efficientnet_b0", side_effect=OSError("simulated download failure")) as factory:
        try:
            create_model(ModelConfig(pretrained=True))
        except RuntimeError as error:
            require("no automatic fallback" in str(error), "Download failure lacks guidance")
        else:
            raise AssertionError("Pretrained download failure was hidden")
        require(factory.call_count == 1, "Unexpected random-weight fallback")
    print("PASS: ImageNet weight selection and download-failure reporting (mocked; no download)", flush=True)


def verify_metric_aggregation() -> None:
    """Unequal batches must be weighted by sample count, including the last batch."""
    logits = torch.tensor([[3.0, 0.0], [0.0, 3.0], [3.0, 0.0]])
    labels = torch.tensor([0, 1, 1])
    metrics = validate(nn.Identity(), [(logits[:2], labels[:2]), (logits[2:], labels[2:])], nn.CrossEntropyLoss(), torch.device("cpu"))
    expected_loss = nn.CrossEntropyLoss()(logits, labels).item()
    require(abs(metrics.loss - expected_loss) < 1e-6, "Loss is not weighted by batch size")
    require(metrics.accuracy == 2 / 3 and metrics.samples == 3, "Wrong accuracy/sample count")
    expect_error(ValueError, lambda: validate(nn.Identity(), [], nn.CrossEntropyLoss(), torch.device("cpu")), "Empty validation loader accepted")
    print("PASS: sample-weighted loss/accuracy with an uneven final batch; empty-loader rejection", flush=True)


def verify_best_selection(model, datasets, loaders, data_config) -> None:
    """Exercise orchestration with controlled metrics and no accessible test loader."""
    train_metrics = [EpochMetrics(1.0, accuracy, 4) for accuracy in (1.0, 0.75, 0.5, 1.0)]
    val_metrics = [EpochMetrics(loss, accuracy, 4) for loss, accuracy in ((2.0, 0.25), (1.5, 0.5), (1.0, 0.5), (0.5, 0.25))]
    development = ({name: datasets[name] for name in ("train", "validation")}, {name: loaders[name] for name in ("train", "validation")})
    with (
        patch.object(train_module, "create_data_pipeline", return_value=development),
        patch.object(train_module, "create_model", return_value=model),
        patch.object(train_module, "train_one_epoch", side_effect=train_metrics) as training,
        patch.object(train_module, "validate", side_effect=val_metrics) as validation,
        patch.object(train_module, "save_checkpoint") as save,
        redirect_stdout(io.StringIO()),
    ):
        history = train_module.run_training(data_config, TrainingConfig(epochs=4), requested_device="cpu", pretrained=False)
        require(training.call_count == validation.call_count == len(history) == 4, "Missing epoch validation")
        require([call.kwargs["training"]["epoch"] for call in save.call_args_list] == [1, 2], "Best selection must ignore ties, train accuracy, and validation loss")
    print("PASS: best checkpoint follows validation accuracy, keeps earliest tie, and needs no test loader (controlled metrics)", flush=True)


def verify_training(config: DataConfig, requested_device: str, pretrained: bool) -> dict:
    verify_runtime()
    verify_pretrained_contract()
    verify_metric_aggregation()
    device = select_device(requested_device)
    seed_training(config.random_seed)
    datasets, loaders = create_data_pipeline(config)
    require(datasets["train"].classes == DEVELOPMENT_CLASSES, "Phase 2 class order changed")
    require(datasets["train"].class_to_idx == datasets["validation"].class_to_idx, "Class mappings disagree")
    images, labels = next(iter(loaders["train"]))
    validation_batch = next(iter(loaders["validation"]))
    model_config = ModelConfig(num_classes=len(datasets["train"].classes), pretrained=pretrained)
    model = create_model(model_config).to(device)
    require(model.classifier[1].out_features == len(DEVELOPMENT_CLASSES), "Classifier has incorrect output count")
    require(all(parameter.requires_grad for parameter in model.parameters()), "Full-network fine-tuning disabled")
    model.eval()
    with torch.no_grad():
        logits = model(images.to(device))
        initial_loss = nn.CrossEntropyLoss()(logits, labels.to(device))
    require(list(logits.shape) == [len(labels), len(DEVELOPMENT_CLASSES)], "Wrong output shape")
    require(torch.isfinite(logits).all().item() and torch.isfinite(initial_loss).item(), "Nonfinite forward/loss")
    print(f"PASS: real Phase 2 batch -> logits {list(logits.shape)} on {device}; finite cross-entropy loss", flush=True)
    training_config = TrainingConfig(epochs=1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=training_config.learning_rate, weight_decay=training_config.weight_decay)
    head_before = model.classifier[1].weight.detach().clone()
    backbone_before = next(model.features.parameters()).detach().clone()
    train_metrics = train_one_epoch(model, [(images, labels)], nn.CrossEntropyLoss(), optimizer, device)
    require(not torch.equal(head_before, model.classifier[1].weight), "Training did not update the head")
    require(not torch.equal(backbone_before, next(model.features.parameters())), "Training did not update the backbone")
    require(all(parameter.grad is None or torch.isfinite(parameter.grad).all().item() for parameter in model.parameters()), "Nonfinite gradients")
    print("PASS: one real-batch backward/AdamW step updates both classifier and backbone", flush=True)
    optimizer.zero_grad(set_to_none=True)
    before_validation = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    validation_calls = []

    def check_validation_mode(module, inputs, output):
        require(not module.training and not torch.is_grad_enabled() and not output.requires_grad, "Validation enables training/gradients")
        validation_calls.append(True)

    hook = model.register_forward_hook(check_validation_mode)
    try:
        validation_metrics = validate(model, [validation_batch], nn.CrossEntropyLoss(), device)
        repeated_metrics = validate(model, [validation_batch], nn.CrossEntropyLoss(), device)
    finally:
        hook.remove()
    require(len(validation_calls) == 2 and validation_metrics == repeated_metrics, "Validation is not deterministic")
    require(all(torch.equal(value, model.state_dict()[name].cpu()) for name, value in before_validation.items()), "Validation changed parameters or BatchNorm buffers")
    require(all(parameter.grad is None for parameter in model.parameters()), "Validation created parameter gradients")
    print("PASS: validation uses eval/no_grad, is repeatable, and preserves all parameters, buffers, and gradients", flush=True)
    training_metadata = {
        "epoch": 1, "config": asdict(training_config), "train_metrics": asdict(train_metrics),
        "validation_metrics": asdict(validation_metrics), "selection_metric": "validation_accuracy",
        "device": str(device), "optimizer": "AdamW", "loss": "CrossEntropyLoss",
    }
    with tempfile.TemporaryDirectory(prefix="food-phase3-") as directory:
        path = Path(directory) / "best.pt"
        save_checkpoint(
            path, model=model, optimizer=optimizer, model_config=model_config,
            data_config=config, class_to_idx=datasets["train"].class_to_idx, training=training_metadata,
        )
        with patch.object(checkpoint_module, "create_model", wraps=create_model) as restore:
            loaded, metadata = load_checkpoint(path, device=device)
            require(restore.call_args.args[0].pretrained is False, "Checkpoint restoration requests ImageNet weights")
        require(metadata["classes"] == list(datasets["train"].classes), "Class order lost")
        require(metadata["class_to_idx"] == datasets["train"].class_to_idx, "Class mapping lost")
        require(metadata["training"] == training_metadata and metadata["model_config"] == asdict(model_config), "Training/model metadata lost")
        require(not loaded.training, "Loaded model must default to eval mode")
        require(all(torch.equal(value, loaded.state_dict()[name].cpu()) for name, value in before_validation.items()), "Model weights/buffers changed in checkpoint round-trip")
        with torch.no_grad():
            torch.testing.assert_close(loaded(validation_batch[0].to(device)), model(validation_batch[0].to(device)))
        restored_optimizer = torch.optim.AdamW(loaded.parameters())
        restored_optimizer.load_state_dict(metadata["optimizer_state_dict"])
        require(len(restored_optimizer.state) == len(optimizer.state) > 0, "Optimizer state was not restored")
        expect_error(ValueError, lambda: load_checkpoint(path, expected_classes=tuple(reversed(DEVELOPMENT_CLASSES))), "Wrong expected class order accepted")
        # Each invalid fixture fails before model reconstruction.
        for key, value in (
            ("format_version", -1), ("model_name", "wrong_model"),
            ("class_to_idx", {}), ("preprocessing", {}),
            ("model_config", {**metadata["model_config"], "num_classes": 9}),
        ):
            torch.save({**metadata, key: value}, path)
            expect_error(ValueError, lambda: load_checkpoint(path), f"Invalid {key} accepted")
        broken_state = dict(metadata["model_state_dict"])
        broken_state.pop("classifier.1.weight")
        torch.save({**metadata, "model_state_dict": broken_state}, path)
        expect_error(RuntimeError, lambda: load_checkpoint(path), "Incomplete model state accepted")
    print("PASS: checkpoint state/logits/optimizer/metadata/class mapping round-trip, offline restore, and incompatible checkpoint rejection", flush=True)
    verify_best_selection(model, datasets, loaders, config)
    return {
        "status": "passed", "device": str(device), "pretrained": pretrained,
        "seed": config.random_seed, "batch_size": config.batch_size, "num_workers": config.num_workers,
        "output_shape": list(logits.shape), "class_to_idx": datasets["train"].class_to_idx,
        "train_samples": len(datasets["train"]), "validation_samples": len(datasets["validation"]),
        "training_steps": 1, "train_metrics": asdict(train_metrics),
        "validation_metrics": asdict(validation_metrics), "test_samples_evaluated": 0,
        "note": "Smoke metrics only; not model-quality estimates. Device branches use mocked availability; model execution uses the reported device.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DataConfig().root)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="cpu")
    parser.add_argument("--pretrained", action="store_true", help="Explicitly allow ImageNet weight download; default is an offline random-weight smoke test")
    parser.add_argument("--report", type=Path, default=PROCESSED_DIR / "phase3_verification.json")
    arguments = parser.parse_args()
    torch.set_num_threads(1)
    report = verify_training(
        DataConfig(root=arguments.root, batch_size=arguments.batch_size, num_workers=arguments.num_workers),
        arguments.device, arguments.pretrained,
    )
    arguments.report.parent.mkdir(parents=True, exist_ok=True)
    arguments.report.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Phase 3 smoke verification passed. Report: {arguments.report}", flush=True)


if __name__ == "__main__":
    main()
