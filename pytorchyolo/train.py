from __future__ import division
import argparse
import tqdm

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from models import load_model
from utils.logger import Logger
from utils.utils import load_classes, worker_seed_set, provide_determinism, to_cpu
from utils.datasets import ListDataset
from utils.augmentations import AUGMENTATION_TRANSFORMS
# from pytorchyolo.utils.transforms import DEFAULT_TRANSFORMS
from utils.parse_config import parse_data_config
from utils.loss import compute_loss
from terminaltables import AsciiTable

from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter 

def _create_data_loader(img_path, batch_size, img_size, n_cpu, multiscale_training=False):
    """Creates a DataLoader for training.

    :param img_path: Path to file containing all paths to training images.
    :type img_path: str
    :param batch_size: Size of each image batch
    :type batch_size: int
    :param img_size: Size of each image dimension for yolo
    :type img_size: int
    :param n_cpu: Number of cpu threads to use during batch generation
    :type n_cpu: int
    :param multiscale_training: Scale images to different sizes randomly
    :type multiscale_training: bool
    :return: Returns DataLoader
    :rtype: DataLoader
    """
    dataset = ListDataset(
        img_path,
        img_size=img_size,
        multiscale=multiscale_training,
        transform=AUGMENTATION_TRANSFORMS)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_cpu,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
        worker_init_fn=worker_seed_set)
    return dataloader


def run():

    parser = argparse.ArgumentParser(description="Trains the YOLO model.")
    parser.add_argument("-m", "--model", type=str, default="config/yolov3.cfg", help="Path to model definition file (.cfg)")
    parser.add_argument("-d", "--data", type=str, default="config/mask.data", help="Path to data config file (.data)")
    parser.add_argument("-e", "--epochs", type=int, default=300, help="Number of epochs")
    parser.add_argument("-v", "--verbose", action='store_true', help="Makes the training more verbose")
    parser.add_argument("--n_cpu", type=int, default=8, help="Number of cpu threads to use during batch generation")
    parser.add_argument("-p", "--pretrained_weights", type=str, help="Path to checkpoint file (.weights or .pth). Starts training from checkpoint model")
    parser.add_argument("--checkpoint_interval", type=int, default=10, help="Interval of epochs between saving model weights")
    parser.add_argument("--evaluation_interval", type=int, default=1, help="Interval of epochs between evaluations on validation set")
    parser.add_argument("--multiscale_training", action="store_true", help="Allow multi-scale training")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="Evaluation: IOU threshold required to qualify as detected")
    parser.add_argument("--conf_thres", type=float, default=0.1, help="Evaluation: Object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.5, help="Evaluation: IOU threshold for non-maximum suppression")
    parser.add_argument("--logdir", type=str, default="logs", help="Directory for training log files (e.g. for TensorBoard)")
    parser.add_argument("--seed", type=int, default=-1, help="Makes results reproducable. Set -1 to disable.")
    args = parser.parse_args()
    print(f"Command line arguements: {args}")

    if args.seed != -1:
        provide_determinism(args.seed)

    logger = Logger(args.logdir)
    
    # Get data configuration
    data_config = parse_data_config(args.data)
    train_path = data_config['train']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create model
    model = load_model(args.model, args.pretrained_weights)

    
    mini_batch_size = model.hyperparams['batch'] // model.hyperparams['subdivisions']
    # mini_batch_size = 2

    # #################
    # Create Dataloader
    # #################

    # Load training dataloader
    dataloader = _create_data_loader(
        train_path,
        mini_batch_size, 
        model.hyperparams['height'],
        args.n_cpu,
        args.multiscale_training        
    )


    ###################
    # Create optimizer 
    ###################
    params = [p for p in model.parameters() if p.requires_grad]
    if (model.hyperparams['optimizer']) in [None, "adam"]:
        optimizer = optim.Adam( 
            params, 
            lr=model.hyperparams['learning_rate'], 
            weight_decay=model.hyperparams['decay']
        )

    elif (model.hyperparams['optimizer'] == 'sgd'):
        optimizer = optim.SGD( 
            params, 
            lr=model.hyperparams['learning_rate'], 
            weight_decay=model.hyperparams['decay'], 
            momentum=model.hyperparams['momentum']
        )
    else:
        print("Unknown optimizer. Please choose between (adam, sgd).")
    
    writer = SummaryWriter()
    for epoch in range(1, args.epochs+1):
        print("\n---------- Training Model ---------")
        model.train()
        for batch_i, (_, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc=f"Training Epoch {epoch}")):
            batches_done = len(dataloader) * epoch + batch_i
            imgs = imgs.to(device, non_blocking=True)
            targets = targets.to(device)
            outputs = model(imgs)
            loss, loss_components = compute_loss(outputs, targets, model)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        writer.add_scalar('Loss/train', loss, epoch)


        # ################
        # Log progress 
        # ################

        if args.verbose:
            print(AsciiTable( 
                [ 
                    ["Type", "Value"], 
                    ["IoU loss", float(loss_components[0])], 
                    ["Object loss", float(loss_components[1])], 
                    ["Class loss", float(loss_components[2])], 
                    ["Loss", float(loss_components[3])], 
                    ["Batch loss", to_cpu(loss).item()],
                ]).table)
        # Tensorboard logging 
        tensorboard_log = [ 
            ("train/iou_loss", float(loss_components[0])), 
            ("train/obj_loss", float(loss_components[1])), 
            ("train/class_loss", float(loss_components[2])), 
            ("train/loss", to_cpu(loss).item())] 
        logger.list_of_scalars_summary(tensorboard_log, batches_done)

        model.seen += imgs.size(0)


        # ##############
        # Save progress 
        # ##############

        # Save model to checkpoint file 
        if epoch % args.checkpoint_interval == 0:
            checkpoint_path = f"checkpoint/yolov3_ckpt_{epoch}.pth"
            print("------ Saving checkpoint to: {} ---".format(checkpoint_path))
            torch.save(model.state_dict(), checkpoint_path)
    

if __name__ == "__main__":
    run()