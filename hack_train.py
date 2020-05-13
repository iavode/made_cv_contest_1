"""Script for baseline training. Model is ResNet18 (pretrained on ImageNet). Training takes ~ 15 mins (@ GTX 1080Ti)."""

import os
import pickle
import sys
from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau # StepLR
import torchvision.models as models
import tqdm
from torch.nn import functional as fnn
from torch.utils import data
from torchvision import transforms

from hack_utils import NUM_PTS, CROP_SIZE
from hack_utils import ScaleMinSideToSize, CropCenter, TransformByKeys
from hack_utils import ThousandLandmarksDataset
from hack_utils import restore_landmarks_batch, create_submission

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def parse_arguments():
    parser = ArgumentParser(__doc__)
    parser.add_argument("--name", "-n",
                        help="Experiment name (for saving checkpoints and submits).",
                        default="baseline")
    parser.add_argument("--data", "-d",
                        help="Path to dir with target images & landmarks.",
                        default=None)
    parser.add_argument("--batch-size", "-b", default=512, type=int)  # 512 is OK for resnet18 finetune @ 6Gb of VRAM
    parser.add_argument("--epochs", "-e", default=1, type=int)
    parser.add_argument("--learning-rate", "-lr", default=1e-2, type=float)
    parser.add_argument("--pretrained-model", "-pm",
                        help="Path to pretrained model",
                        default=None, type=str),
    parser.add_argument("--gpu", action="store_true")
    return parser.parse_args()


def train(model, loader, loss_fn, optimizer, device):
    model.train()
    train_loss = []
    for batch in tqdm.tqdm(loader, total=len(loader), desc="training..."):
        images = batch["image"].to(device)  # B x 3 x CROP_SIZE x CROP_SIZE
        landmarks = batch["landmarks"]  # B x (2 * NUM_PTS)

        pred_landmarks = model(images).cpu()  # B x (2 * NUM_PTS)
        loss = loss_fn(pred_landmarks, landmarks, reduction="mean")
        train_loss.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return np.mean(train_loss)


def validate(model, loader, loss_fn, device):
    model.eval()
    val_loss = []
    for batch in tqdm.tqdm(loader, total=len(loader), desc="validation..."):
        images = batch["image"].to(device)
        landmarks = batch["landmarks"]

        with torch.no_grad():
            pred_landmarks = model(images).cpu()
        loss = loss_fn(pred_landmarks, landmarks, reduction="mean")
        val_loss.append(loss.item())

    return np.mean(val_loss)


def predict(model, loader, device):
    model.eval()
    predictions = np.zeros((len(loader.dataset), NUM_PTS, 2))
    for i, batch in enumerate(tqdm.tqdm(loader, total=len(loader), desc="test prediction...")):
        images = batch["image"].to(device)

        with torch.no_grad():
            pred_landmarks = model(images).cpu()
        pred_landmarks = pred_landmarks.numpy().reshape((len(pred_landmarks), NUM_PTS, 2))  # B x NUM_PTS x 2

        fs = batch["scale_coef"].numpy()  # B
        margins_x = batch["crop_margin_x"].numpy()  # B
        margins_y = batch["crop_margin_y"].numpy()  # B
        prediction = restore_landmarks_batch(
            pred_landmarks, fs, margins_x, margins_y)  # B x NUM_PTS x 2
        predictions[i * loader.batch_size: (i + 1) * loader.batch_size] = prediction

    return predictions


def main(args):
    # 1. prepare data & models
    # применение новых трансформаций не дало улучшения результатов
    # единственное изменение это параметры нормализации
    train_transforms = transforms.Compose([
        ScaleMinSideToSize((CROP_SIZE, CROP_SIZE)),
        CropCenter(CROP_SIZE),
        TransformByKeys(transforms.ToPILImage(), ("image",)),
        TransformByKeys(transforms.ToTensor(), ("image",)),
        TransformByKeys(
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # средние значения и дисперсии взяты из документации pytorch,
                std=[0.229, 0.224, 0.225]),  # с такими же значениями обучалась сеть на корпусе imagenet
            ("image",),
        ),
    ])
    device = torch.device("cuda: 0") if args.gpu else torch.device("cpu")
    print("Creating model...")
    model = models.resnext50_32x4d(pretrained=True)
    in_features = model.fc.in_features
     fc = nn.Sequential(
         nn.Linear(in_features, 2 * NUM_PTS),) # новая "голова"
     model.fc = fc
     state_dict = None
     #  если есть сеть дообученная на датасете из контеста
     if args.pretrained_model:
         print(f"Load best_state_dict {args.pretrained_model}")
         state_dict = torch.load(args.pretrained_model)
         model.load_state_dict(state_dict)
         del state_dict

    model.to(device)
    print(model)

    factor = 0.1**(1/2)
    # оптимизатора выбран AdamW, с небольшой нормализацией весов
    optimizer = optim.AdamW(
        model.parameters(), lr=args.learning_rate,
        amsgrad=True, weight_decay=0.05
    )
    loss_fn = fnn.mse_loss
    # изменения lr происходит при помощи ReduceLROnPlateau
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', patience=1, factor=factor,
    )

    print(loss_fn)
    print(optimizer)
    print(scheduler)


     print("Reading data...")
     print("Read train landmark dataset")
     train_dataset = ThousandLandmarksDataset(
         os.path.join(args.data, 'train'), train_transforms, split="train")
     print("Create picture loader for test dataset")
     train_dataloader = data.DataLoader(
         train_dataset, batch_size=args.batch_size,
         num_workers=0, pin_memory=True,
         shuffle=True, drop_last=True)
     print("Read val landmark dataset")
     val_dataset = ThousandLandmarksDataset(
         os.path.join(args.data, 'train'),
         train_transforms, split="val"
     )
     print("Create picture loader for val dataset")
     val_dataloader = data.DataLoader(
         val_dataset, batch_size=args.batch_size,
         num_workers=0, pin_memory=True,
         shuffle=False, drop_last=False
     )

     # 2. train & validate
     print("Ready for training...")
     best_val_loss = np.inf
     for epoch in range(args.epochs):
         train_loss = train(
             model, train_dataloader, loss_fn, optimizer, device=device)
         val_loss = validate(
             model, val_dataloader, loss_fn, device=device)
         print(
             "Epoch #{:2}:\ttrain loss: {:5.5}\tval loss: {:5.5}".format(
                 epoch + 1, train_loss, val_loss)
         )
         scheduler.step(val_loss)
         if val_loss < best_val_loss:
             best_val_loss = val_loss
             with open(f"{args.name}_best.pth", "wb") as fp:
                 torch.save(model.state_dict(), fp)

    # 3. predict
    print("Start predict")
    test_dataset = ThousandLandmarksDataset(
        os.path.join(args.data, 'test'), train_transforms, split="test")
    test_dataloader = data.DataLoader(
        test_dataset, batch_size=args.batch_size,
        num_workers=0, pin_memory=True,
        shuffle=False, drop_last=False
    )

    with open(f"{args.name}_best.pth", "rb") as fp:
        best_state_dict = torch.load(fp, map_location="cpu")
        model.load_state_dict(best_state_dict)

    test_predictions = predict(model, test_dataloader, device)
    with open(f"{args.name}_test_predictions.pkl", "wb") as fp:
        pickle.dump({"image_names": test_dataset.image_names,
                     "landmarks": test_predictions}, fp)

    create_submission(args.data, test_predictions, f"{args.name}_submit.csv")


if __name__ == '__main__':
    args = parse_arguments()
    sys.exit(main(args))
