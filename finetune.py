import os
from utils import find_free_gpu

os.environ["CUDA_VISIBLE_DEVICES"] = find_free_gpu()

import copy
import json
import wandb
import torch
import argparse
import pandas as pd
from spice import SPICE
from datetime import datetime
import pytorch_lightning as pl
from shapetalk import ShapeTalk
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

torch.set_float32_matmul_precision(precision="high")

BASE_DIR = "/scratch/noam/control_point_e"
PARTNET_DICTS_DIR = "/scratch/noam/partnet"
LLAMA3_WNLEMMA_UTTERANCE = "llama3_wnlemma_utterance"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument("--val_freq", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--object", type=str, default="chair")
    parser.add_argument("--subset_size", type=int, default=500)
    parser.add_argument("--copy_prob", type=float, default=0.1)
    parser.add_argument("--copy_prompt", type=str, default="COPY")
    parser.add_argument("--num_val_samples", type=int, default=10)
    parser.add_argument("--num_test_samples", type=int, default=10)
    parser.add_argument("--cond_drop_prob", type=float, default=0.1)
    parser.add_argument("--wandb_project", type=str, default="SPICE")
    return parser.parse_args()


def build_name(args):
    name = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
    name += f"_{args.object}"
    name += f"_copy_{args.copy_prob}_'{args.copy_prompt}'"
    name += f"_cond_drop_{args.cond_drop_prob}"
    if args.subset_size is not None:
        name += f"_subset_{args.subset_size}"
    return name


def build_dataset(partnet_dict, subset_size, data_csv, device, batch_size):
    df = pd.read_csv(os.path.join(BASE_DIR, "datasets", data_csv))
    df = df[df[LLAMA3_WNLEMMA_UTTERANCE] != "Unknown"]
    df = df[df.source_uid.apply(lambda uid: uid in partnet_dict)]
    df = df[df.target_uid.apply(lambda uid: uid in partnet_dict)]
    df.sort_values("chamfer_distance")
    if subset_size is not None and subset_size < len(df):
        df = df.head(subset_size)
    return ShapeTalk(
        df=df,
        device=device,
        batch_size=batch_size,
        partnet_dict=partnet_dict,
    )


def main(args):
    name = build_name(args)
    output_dir = os.path.join(BASE_DIR, "executions", name)
    os.makedirs(output_dir, exist_ok=True)
    os.environ["WANDB_DIR"] = os.path.join(output_dir)
    os.environ["WANDB_API_KEY"] = "7b14a62f11dc360ce036cf59b53df0c12cd87f5a"
    wandb.init(project=args.wandb_project, name=name, config=vars(args))
    with open(os.path.join(PARTNET_DICTS_DIR, f"{args.object}.json")) as f:
        partnet_dict = json.load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = build_dataset(
        device=device,
        partnet_dict=partnet_dict,
        batch_size=args.batch_size,
        subset_size=args.subset_size,
        data_csv=os.path.join(args.object, "train.csv"),
    )
    train_dataloader = DataLoader(
        shuffle=True,
        dataset=train_dataset,
        batch_size=args.batch_size,
    )
    val_dataset = copy.deepcopy(train_dataset)
    val_dataset.set_length(
        length=args.num_val_samples,
        batch_size=args.num_val_samples,
    )
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=args.num_val_samples)
    test_dataset = build_dataset(
        device=device,
        partnet_dict=partnet_dict,
        batch_size=args.num_test_samples,
        subset_size=args.num_test_samples,
        data_csv=os.path.join(args.object, "val.csv"),
    )
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=args.num_test_samples)
    model = SPICE(
        lr=args.lr,
        dev=device,
        copy_prob=args.copy_prob,
        batch_size=args.batch_size,
        copy_prompt=args.copy_prompt,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        cond_drop_prob=args.cond_drop_prob,
    )
    wandb.watch(model)
    checkpoint_callback = ModelCheckpoint(
        save_top_k=-1,
        save_weights_only=True,
        every_n_epochs=args.val_freq,
        dirpath=os.path.join(output_dir, "checkpoints"),
    )
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accumulate_grad_batches=10,
        callbacks=[checkpoint_callback],
        logger=TensorBoardLogger(output_dir),
        check_val_every_n_epoch=args.val_freq,
    )
    trainer.fit(
        model=model,
        val_dataloaders=val_dataloader,
        train_dataloaders=train_dataloader,
    )
    if test_dataloader:
        trainer.test(model=model, dataloaders=test_dataloader)
    wandb.finish()


if __name__ == "__main__":
    main(args=parse_args())
