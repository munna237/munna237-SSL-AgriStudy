
import os, sys, json, argparse
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, TensorDataset

# Paths must be inside script for subprocess calls
sys.path.insert(0, "/content/SSL_AgriStudy/dino")
sys.path.insert(0, "/content/SSL_AgriStudy/mae")


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",       required=True,
        choices=["dino_vits16","dinov2_vits14",
                 "dinov2_vitb14","mae_vitb16"])
    p.add_argument("--checkpoint",  required=True)
    p.add_argument("--data_path",   required=True)
    p.add_argument("--output_dir",  required=True)
    p.add_argument("--protocol",    required=True,
        choices=["linear","finetune"])
    p.add_argument("--num_classes", type=int, required=True)
    p.add_argument("--epochs",      type=int, default=100)
    p.add_argument("--batch_size",  type=int, default=128)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed",        type=int, default=42)
    return p.parse_args()


def load_backbone(model_name, ckpt_path):
    """Load backbone and return (model, embed_dim)."""
    if model_name == "dino_vits16":
        import vision_transformer as vits
        m   = vits.vit_small(patch_size=16)
        ckpt = torch.load(ckpt_path, map_location="cpu",
                          weights_only=False)
        sd  = ckpt.get("teacher",
              ckpt.get("model", ckpt))
        sd  = {k.replace("module.","")
                .replace("backbone.",""): v
               for k, v in sd.items()
               if "head"       not in k
               and "mlp"       not in k
               and "last_layer"not in k}
        m.load_state_dict(sd, strict=False)
        dim = 384

    elif model_name == "dinov2_vits14":
        m   = torch.hub.load("facebookresearch/dinov2",
                              "dinov2_vits14")
        dim = 384

    elif model_name == "dinov2_vitb14":
        m   = torch.hub.load("facebookresearch/dinov2",
                              "dinov2_vitb14")
        dim = 768

    elif model_name == "mae_vitb16":
        import models_mae
        m    = models_mae.mae_vit_base_patch16()
        ckpt = torch.load(ckpt_path, map_location="cpu",
                          weights_only=False)
        sd   = ckpt.get("model", ckpt)
        m.load_state_dict(sd, strict=False)
        dim  = 768

    m.eval()
    return m, dim


def get_transforms():
    norm = T.Normalize([0.485,0.456,0.406],
                       [0.229,0.224,0.225])
    train_tf = T.Compose([
        T.RandomResizedCrop(224, scale=(0.08,1.0)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(0.4,0.4,0.4,0.1),
        T.ToTensor(), norm])
    val_tf = T.Compose([
        T.Resize(256), T.CenterCrop(224),
        T.ToTensor(), norm])
    return train_tf, val_tf


def get_features(backbone, loader, device, model_name):
    """Extract frozen features from backbone."""
    backbone.eval()
    feats_all, labels_all = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            if model_name == "mae_vitb16":
                lat, _, _ = backbone.forward_encoder(
                    imgs, mask_ratio=0.0)
                f = lat[:, 0, :]
            else:
                f = backbone(imgs)
            feats_all.append(f.cpu())
            labels_all.append(labels)
    return (torch.cat(feats_all,  dim=0),
            torch.cat(labels_all, dim=0))


def linear_probe(backbone, dim, n_cls,
                 tr_loader, val_loader,
                 device, args, log_path):
    print("  Extracting features (train)...")
    tr_X,  tr_y  = get_features(
        backbone, tr_loader,  device, args.model)
    print("  Extracting features (val)...")
    val_X, val_y = get_features(
        backbone, val_loader, device, args.model)

    clf  = nn.Linear(dim, n_cls).to(device)
    opt  = torch.optim.AdamW(
        clf.parameters(), lr=1e-3, weight_decay=1e-4)
    sch  = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs)
    loss_fn = nn.CrossEntropyLoss()

    tr_dl  = DataLoader(TensorDataset(tr_X, tr_y),
                        batch_size=256, shuffle=True)
    val_dl = DataLoader(TensorDataset(val_X, val_y),
                        batch_size=256, shuffle=False)

    best, logs = 0.0, []
    for ep in range(args.epochs):
        clf.train()
        for xb, yb in tr_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss_fn(clf(xb), yb).backward()
            opt.step()
        sch.step()

        if (ep+1) % 10 == 0 or ep == args.epochs-1:
            clf.eval()
            correct = total = 0
            with torch.no_grad():
                for xb, yb in val_dl:
                    preds = clf(xb.to(device)).argmax(1).cpu()
                    correct += (preds == yb).sum().item()
                    total   += len(yb)
            acc  = 100.0 * correct / total
            best = max(best, acc)
            logs.append({"epoch": ep+1, "test_acc1": acc})
            print(f"  Ep {ep+1:3d}: {acc:.2f}%"
                  f"  best={best:.2f}%")

    with open(log_path, "w") as f:
        [f.write(json.dumps(l)+"\n") for l in logs]
    print(f"  ✅ Linear probe best: {best:.2f}%")
    return best


def finetune(backbone, dim, n_cls,
             tr_loader, val_loader,
             device, args, log_path):
    is_mae = (args.model == "mae_vitb16")

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.bb   = backbone
            self.head = nn.Linear(dim, n_cls)
        def forward(self, x):
            if is_mae:
                lat, _, _ = self.bb.forward_encoder(
                    x, mask_ratio=0.0)
                f = lat[:, 0, :]
            else:
                f = self.bb(x)
            return self.head(f)

    model = Model().to(device)
    opt   = torch.optim.AdamW([
        {"params": backbone.parameters(), "lr": 1e-5},
        {"params": model.head.parameters(),"lr": 1e-4},
    ], weight_decay=1e-4)
    sch     = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    best, logs = 0.0, []
    for ep in range(args.epochs):
        model.train()
        for imgs, labels in tr_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            opt.zero_grad()
            loss_fn(model(imgs), labels).backward()
            opt.step()
        sch.step()

        if (ep+1) % 5 == 0 or ep == args.epochs-1:
            model.eval()
            correct = total = 0
            with torch.no_grad():
                for imgs, labels in val_loader:
                    preds = model(
                        imgs.to(device)).argmax(1).cpu()
                    correct += (preds == labels).sum().item()
                    total   += len(labels)
            acc  = 100.0 * correct / total
            best = max(best, acc)
            logs.append({"epoch": ep+1, "test_acc1": acc})
            print(f"  Ep {ep+1:3d}: {acc:.2f}%"
                  f"  best={best:.2f}%")

    with open(log_path, "w") as f:
        [f.write(json.dumps(l)+"\n") for l in logs]
    print(f"  ✅ Finetune best: {best:.2f}%")
    return best


def main():
    args   = get_args()
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n{'='*50}")
    print(f"Model:    {args.model}")
    print(f"Dataset:  {os.path.basename(args.data_path)}")
    print(f"Protocol: {args.protocol}")
    print(f"Classes:  {args.num_classes}")
    print(f"Device:   {device}")
    print(f"{'='*50}")

    backbone, dim = load_backbone(
        args.model, args.checkpoint)
    backbone = backbone.to(device)
    print(f"✅ Model loaded  dim={dim}")

    train_tf, val_tf = get_transforms()

    # Handle datasets with val/ instead of test/
    train_path = os.path.join(args.data_path, "train")
    test_path  = os.path.join(args.data_path, "test")
    if not os.path.exists(test_path):
        test_path = os.path.join(args.data_path, "val")
        print("Using val/ as test split")

    tr_ds  = datasets.ImageFolder(train_path, train_tf, allow_empty=True)
    val_ds = datasets.ImageFolder(test_path,  val_tf, allow_empty=True)
    tr_loader  = DataLoader(
        tr_ds,  batch_size=args.batch_size,
        shuffle=True,  num_workers=args.num_workers,
        pin_memory=True)
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers,
        pin_memory=True)
    print(f"Train: {len(tr_ds):,}  Test: {len(val_ds):,}")

    log_path = os.path.join(
        args.output_dir,
        f"{args.model}_{args.protocol}_log.txt")

    if args.protocol == "linear":
        acc = linear_probe(
            backbone, dim, args.num_classes,
            tr_loader, val_loader,
            device, args, log_path)
    else:
        acc = finetune(
            backbone, dim, args.num_classes,
            tr_loader, val_loader,
            device, args, log_path)

    # Save result
    result = {
        "model":       args.model,
        "dataset":     os.path.basename(args.data_path),
        "protocol":    args.protocol,
        "num_classes": args.num_classes,
        "best_acc":    acc,
    }
    rpath = os.path.join(
        args.output_dir,
        f"{args.model}_{args.protocol}_result.json")
    with open(rpath, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n✅ FINAL RESULT: {acc:.2f}%")


if __name__ == "__main__":
    main()
