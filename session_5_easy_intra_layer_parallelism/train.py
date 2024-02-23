import torch
import torchvision
import sys
import os
import time
import json
from torchvision import transforms
import numpy as np
from axonn import axonn as ax

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from model.fc_net_easy_tensor_parallel import FC_Net
from utils import print_memory_stats, num_params, log_dist
from args import create_parser

NUM_EPOCHS=10
PRINT_EVERY=200

torch.device('cpu')
torch.manual_seed(0)

if __name__ == "__main__":
    data_json = {
        "iter_times": [],
        "iter_losses": [],
        "val_acc": [],
        "val_loss": [],
        "train_acc": [],
        "train_loss": []
    }
    
    parser = create_parser()
    args = parser.parse_args()

    ## Step 1 - Initialize AxoNN
    ax.init(
                G_data=args.G_data,
                G_inter=1,
                G_intra_r=args.G_intra_r,
                G_intra_c=args.G_intra_c,
                mixed_precision=False,
                fp16_allreduce=False,
                device = 'cpu',
            )

    log_dist('initialized AxoNN', ranks=[0])

    augmentations = transforms.Compose(
        [
            transforms.Resize(args.image_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )

    ## Step 2 - Create dataset with augmentations
    train_dataset = torchvision.datasets.MNIST(
        root=args.data_dir, train=True, transform=augmentations
    )

    ## Step 3 - Create dataloader using AxoNN
    train_loader = ax.create_dataloader(
        train_dataset,
        args.batch_size,
        args.micro_batch_size,
        num_workers=1,
    )

    ## Step 4 - Create Neural Network 
    net = FC_Net(args.num_layers, args.image_size**2, args.hidden_size, 10).to(torch.device('cpu'))
    params = num_params(net) / 1e9

    ## Step 5 - Create Optimizer 
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    ## Step 6 - register model and optimizer with AxoNN
    ## This creates the required data structures for
    ## mixed precision
    net, optimizer = ax.register_model_and_optimizer(net, optimizer)

    ## Step 7 - Create Loss Function and register it
    loss_fn = torch.nn.CrossEntropyLoss()
    ax.register_loss_fn(loss_fn)

    ## Step 8 - Train
    log_dist(f"Model Params = {num_params(net)*ax.config.G_intra/1e9} B", [0])
    log_dist(f"Start Training with AxoNN's Intra-Layer Parallelism", [0])

    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0
        total = 0
        correct = 0
        iter_ = 0
        iter_times = []
        iter_losses = []
        for img, label in train_loader:
            start_time_iter = time.time()
            optimizer.zero_grad()
            
            output = net(img)
            img = img.to(torch.device('cpu'))
            label = label.to(torch.device('cpu'))
            iter_loss = ax.run_batch(img, label)
            iter_losses.append(iter_loss)
            optimizer.step() 

            epoch_loss += iter_loss
            _, predicted = output.max(1)
            total += label.size(0)
            correct += predicted.eq(label).sum().item()
            iter_time = time.time() - start_time_iter
            iter_times.append(iter_time)
            if iter_ % PRINT_EVERY == 0:
                log_dist(f"Epoch {epoch} | Iter {iter_}/{len(train_loader)} | Iter Train Loss = {iter_loss:.3f} | Iter Time = {iter_time:.6f} s", [0])
            iter_ += 1
        data_json["iter_losses"].append(iter_losses)    
        data_json["train_acc"].append(100.*correct/total)
        data_json["train_loss"].append(epoch_loss/len(train_loader))
        data_json["iter_times"].append(iter_times)
        print_memory_stats()
        log_dist(f"Epoch {epoch} : Epoch Train Loss= {epoch_loss/len(train_loader):.3f} | Average Iter Time = {np.mean(iter_times):.6f} s", [0])
        json.dump(data_json, open("file.json", 'w'), indent=2)
        log_dist(f"End Training ...", [0])
