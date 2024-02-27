import pandas as pd
import torch
import time
import argparse
# import torchvision
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from torch.utils.data import DataLoader
from utils.dataloader import reindex_df, rename_images, get_transform, JudgementDataset
from utils.model import get_pretrained_resnet, WRM
from utils.trainer import Trainer


NUM_EPOCHS = 50
LR = 1e-3
EVAL_PER_EPOCH = 5
GRADIENT_ACCUMULATION_STEPS = 8 #update as a batch of 32 (4*8)

def main(train, test, val, resnet, batch_size, num_workers, num_epoch, save_model_prefix, save_df):

    train_judgements = pd.read_csv(train)
    test_judgements = pd.read_csv(test)
    val_judgements = pd.read_csv(val)
    transform = get_transform()
    train_set = JudgementDataset(judgements=train_judgements, transform=transform, five_crop=True)
    test_set = JudgementDataset(judgements=test_judgements, transform=transform, five_crop=True)
    val_set = JudgementDataset(judgements=val_judgements, transform=get_transform(evaluation=True), five_crop=True, return_index=True)

    train_loader = DataLoader(
        dataset= train_set,
        batch_size= batch_size,
        num_workers= num_workers,
        shuffle= True,
        pin_memory= True,
    )

    test_loader = DataLoader(
        dataset= test_set,
        batch_size= batch_size,
        num_workers= num_workers,
        shuffle= True,
        pin_memory= True,
    )

    val_loader = DataLoader(
        dataset= val_set, 
        batch_size= batch_size, 
        num_workers= num_workers, 
        shuffle= True, 
        pin_memory= True
    )

    device = torch.device("cuda")
    feature = get_pretrained_resnet(device, resnet)
    model = WRM(global_feature=feature, patch_feature=feature, device=device)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr = LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.5, step_size=10)
    writer = SummaryWriter()
    trainer = Trainer(
        optimizer = optimizer,
        scheduler = scheduler,
        writer = writer,
        train_loader = train_loader,
        test_loader = test_loader,
        val_loader = val_loader,
        device = device,
        eval_per_epoch = 5,
        gradient_accumulation = 8,
        save_model_prefix = save_model_prefix
    )
    model = trainer.train(model, num_epoch)
    torch.save(model.state_dict(), f"{save_model_prefix}_last.pth")
    print("==================== Training finished. Model saved as models/walkability_last.pth ====================")
    print(" Evaluation")
    ##### Evaluation ######
    trainer.evaluate(model, val_df = val_judgements)
    val_judgements.to_csv(save_df, index = False)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", help = "file path of the training dataset")
    parser.add_argument("--test", help = "file path of the test dataset")
    parser.add_argument("--val", help = "file path of the validation dataset")
    parser.add_argument("--resnet", help = "file path of pretrained_resnet")
    parser.add_argument("-b", "--batch_size", default=4)
    parser.add_argument("-w", "--num_workers", default=4)
    parser.add_argument("-e", "--num_epoch", default=50)
    parser.add_argument("--save_model_prefix", default="/models/walkability")
    parser.add_argument("--save_df", default="/data/walkability.csv")

    args = parser.parse_args()
    main(args)