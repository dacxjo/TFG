import torch
import lightning as L
from src.modeling.classifier import MolecularSubtypeClassifier
from sklearn.metrics import classification_report, confusion_matrix

from src.utils import get_device


def test_from_checkpoint(dataset, backbone_fn, checkpoint_dir, batch_size=32, num_classes=4, generate_report=False, class_names=None):

    test_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        persistent_workers=True
    )

    model = MolecularSubtypeClassifier.load_from_checkpoint(
        checkpoint_dir,
        backbone=backbone_fn(),
        num_classes=num_classes,
        map_location=get_device()
    )

    model.eval()

    trainer = L.Trainer(
        accelerator='auto',
        enable_model_summary=False
    )

    results = trainer.test(model, test_loader)

    if generate_report:
        all_preds = []
        all_targets = []

        device = next(model.parameters()).device

        with torch.no_grad():
            for batch in test_loader:
                x, y = batch
                x = x.to(device)
                outputs = model(x)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_targets.extend(y.numpy())

        if class_names is None:
            class_names = [f"Clase {i}" for i in range(num_classes)]

        report = classification_report(
            all_targets,
            all_preds,
            target_names=class_names,
            digits=4
        )

        cm = confusion_matrix(all_targets, all_preds)

        print("Classification Report:")
        print(report)
        print("Confusion Matrix:")
        print(cm)

    return results

def test_from_weights(dataset, backbone, weights_path, batch_size=32, num_classes=4):
    test_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        persistent_workers=True
    )

    model = MolecularSubtypeClassifier(
        backbone=backbone,
        num_classes=num_classes,
    )

    model.load_state_dict(torch.load(weights_path))

    model.eval()

    trainer = L.Trainer(
        accelerator='auto',
        enable_model_summary=False
    )

    results = trainer.test(model, test_loader)
    return results

