from pathlib import Path
import time
import torch
import numpy as np
import math
import gc
from functools import partial
from dataset_CL import LinearCLDataset
from torch.utils.data import DataLoader
from transformer_onestep import GPTConfig, GPT, warmup_cosine_lr
from transformer_onestep_CL import GPTClosedLoop
import tqdm
import argparse
import warnings


# Disable all user warnings
warnings.filterwarnings("ignore")

# Your code goes here

# Re-enable user warnings
warnings.filterwarnings("default")


def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for batch in dataloader:
        batch_G, batch_r, batch_y_d = batch
        batch_r, batch_y_d = batch_r.to(device), batch_y_d.to(device)

        optimizer.zero_grad()
        batch_y = model(batch_G, batch_r)
        loss = criterion(batch_y, batch_y_d)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            batch_G, batch_r, batch_y_d = batch
            batch_r, batch_y_d = batch_r.to(device), batch_y_d.to(device)

            batch_y = model(batch_G, batch_r)
            loss = criterion(batch_y, batch_y_d)

            running_loss += loss.item()

    return running_loss / len(dataloader)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Meta system identification with transformers')

    # Overall
    parser.add_argument('--model-dir', type=str, default="out", metavar='S',
                        help='Saved model folder')
    parser.add_argument('--out-file', type=str, default="ckpt_onestep_gen_lin_2.1", metavar='S',
                        help='Saved model name')
    parser.add_argument('--in-file', type=str, default="ckpt_onestep_gen_lin_2", metavar='S',
                        help='Loaded model name (when resuming)')
    parser.add_argument('--init-from', type=str, default="pretrained", metavar='S',
                        help='Init from (scratch|resume|pretrained)')
    parser.add_argument('--seed', type=int, default=42, metavar='N',
                        help='Seed for random number generation')
    parser.add_argument('--log-wandb', action='store_true', default=False,
                        help='disables CUDA training')

    # Dataset
    parser.add_argument('--nx', type=int, default=2, metavar='N',
                        help='model order (default: 5)')
    parser.add_argument('--nu', type=int, default=1, metavar='N',
                        help='model order (default: 5)')
    parser.add_argument('--ny', type=int, default=1, metavar='N',
                        help='model order (default: 5)')
    parser.add_argument('--seq-len', type=int, default=500, metavar='N',
                        help='sequence length (default: 600)')
    parser.add_argument('--mag_range', type=tuple, default=(0.5, 0.97), metavar='N',
                        help='sequence length (default: 600)')
    parser.add_argument('--phase_range', type=tuple, default=(0.0, math.pi/2), metavar='N',
                        help='sequence length (default: 600)')
    parser.add_argument('--fixed-system', action='store_true', default=False,
                        help='If True, keep the same model all the times')

    # Model
    parser.add_argument('--n-layer', type=int, default=8, metavar='N',
                        help='number of iterations (default: 1M)')
    parser.add_argument('--n-head', type=int, default=4, metavar='N',
                        help='number of iterations (default: 1M)')
    parser.add_argument('--n-embd', type=int, default=32, metavar='N',
                        help='number of iterations (default: 1M)')
    parser.add_argument('--dropout', type=float, default=0, metavar='LR',
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--bias', action='store_true', default=False,
                        help='bias in model')
    parser.add_argument('--reg_u_weight', type=float, default=0.0, metavar='N',
                        help='bias in model')
    parser.add_argument('--use_p', type=bool, default=True, metavar='N',
                        help='bias in model')
    parser.add_argument('--use_i', type=bool, default=True, metavar='N',
                        help='bias in model')
    parser.add_argument('--use_d', type=bool, default=True, metavar='N',
                        help='bias in model')


    # Training
    parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                        help='batch size (default:32)')
    parser.add_argument('--max-iters', type=int, default=1_000_000, metavar='N',
                        help='number of iterations (default: 1M)')
    parser.add_argument('--warmup-iters', type=int, default=5_000, metavar='N',
                        help='number of iterations (default: 1000)')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--weight-decay', type=float, default=0.0, metavar='D',
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--eval-interval', type=int, default=10, metavar='N',
                        help='batch size (default:32)')
    parser.add_argument('--eval-iters', type=int, default=10, metavar='N',
                        help='batch size (default:32)')
    parser.add_argument('--fixed-lr', action='store_true', default=False,
                        help='disables CUDA training')

    # Compute
    parser.add_argument('--threads', type=int, default=16,
                        help='number of CPU threads (default: 10)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--cuda-device', type=str, default="cuda:0", metavar='S',
                        help='cuda device (default: "cuda:0")')
    parser.add_argument('--compile', action='store_true', default=False,
                        help='disables CUDA training')

    cfg = parser.parse_args()

    # Other settings
    cfg.beta1 = 0.9
    cfg.beta2 = 0.95

    print(cfg.seq_len)

    # Derived settings
    n_skip = 0
    cfg.block_size = cfg.seq_len
    cfg.lr_decay_iters = cfg.max_iters
    cfg.min_lr = cfg.lr/10.0  #
    cfg.decay_lr = not cfg.fixed_lr
    cfg.eval_batch_size = cfg.batch_size

    # Set seed for reproducibility
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed) # not needed? All randomness now handled with generators

    # Create out dir
    model_dir = Path(cfg.model_dir)
    model_dir.mkdir(exist_ok=True)

    # Configure compute
    cuda_device = "cuda:0"
    torch.set_num_threads(cfg.threads)
    use_cuda = not cfg.no_cuda and torch.cuda.is_available()
    device_name = cuda_device if use_cuda else "cpu"
    device = torch.device(device_name)
    device_type = 'cuda' if 'cuda' in device_name else 'cpu' # for later use in torch.autocast
    torch.set_float32_matmul_precision("high")

    print(torch.cuda.is_available())
    print(torch.cuda.current_device())

    train_ds = LinearCLDataset(seq_len=cfg.seq_len, ts=0.01, seed=42, perturb_percentage= 50)
    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size)

    # if we work with a constant model we also validate with the same (thus same seed!)
    val_ds = LinearCLDataset(seq_len=cfg.seq_len, ts=0.01, seed=42, perturb_percentage = 50)
    val_dl = DataLoader(val_ds, batch_size=cfg.eval_batch_size)

    # Model
    model_args = dict(n_layer=cfg.n_layer, n_head=cfg.n_head, n_embd=cfg.n_embd, n_x=cfg.nx, n_y=cfg.ny, n_u=cfg.nu, block_size=cfg.block_size,
                      bias=cfg.bias, dropout=cfg.dropout, use_p=cfg.use_p, use_i=cfg.use_i, use_d=cfg.use_d)  # start with model_args from command line

    if cfg.init_from == "scratch":
        gptconf = GPTConfig(**model_args)
        model = GPTClosedLoop(gptconf)
    elif cfg.init_from == "resume" or cfg.init_from == "pretrained":
        ckpt_path = model_dir / f"{cfg.in_file}.pt"
        checkpoint = torch.load(ckpt_path, map_location=device)
        gptconf = GPTConfig(**checkpoint["model_args"])
        model = GPTClosedLoop(gptconf)
        state_dict = checkpoint['model']
        # fix the keys of the state dictionary :(
        # honestly no idea how checkpoints sometimes get this prefix, have to debug more
        unwanted_prefix = '_orig_mod.'
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
    model.to(device)

    if cfg.compile:
        model = torch.compile(model)  # requires PyTorch 2.0

    # Optimizer
    optimizer = model.configure_optimizers(cfg.weight_decay, cfg.lr, (cfg.beta1, cfg.beta2), device_type)

    if cfg.init_from == "resume":
        optimizer.load_state_dict(checkpoint['optimizer'])

    # Criterion
    criterion = torch.nn.MSELoss()

    # Training and validation loop
    LOSS_ITR = []
    LOSS_VAL = []
    best_val_loss = float('inf')

    if cfg.init_from == ("scrat"
                         "ch") or cfg.init_from == "pretrained":
        iter_num = 0
        best_val_loss = np.inf
    elif cfg.init_from == "resume":
        iter_num = checkpoint["iter_num"]
        best_val_loss = checkpoint['best_val_loss']

    get_lr = partial(warmup_cosine_lr, lr=cfg.lr, min_lr=cfg.min_lr,
                     warmup_iters=cfg.warmup_iters, lr_decay_iters=cfg.lr_decay_iters)
    time_start = time.time()

    for epoch in range(cfg.max_iters):

        if cfg.decay_lr:
            lr_iter = get_lr(epoch)
        else:
            lr_iter = cfg.lr
        optimizer.param_groups[0]['lr'] = lr_iter

        train_loss = train(model, train_dl, criterion, optimizer, device)
        val_loss = validate(model, val_dl, criterion, device)

        LOSS_ITR.append(train_loss)
        LOSS_VAL.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'model_args': model_args,
                'iter_num': epoch,
                'train_time': time.time() - time_start,
                'LOSS': LOSS_ITR,
                'LOSS_VAL': LOSS_VAL,
                'best_val_loss': best_val_loss,
                'cfg': cfg,
            }
            torch.save(checkpoint, model_dir / f"{cfg.out_file}.pt")

        if ( epoch > 0 ) and ( epoch % 10 == 0):
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'model_args': model_args,
                'iter_num': epoch,
                'train_time': time.time() - time_start,
                'LOSS': LOSS_ITR,
                'LOSS_VAL': LOSS_VAL,
                'best_val_loss': best_val_loss,
                'cfg': cfg,
            }
            torch.save(checkpoint, model_dir / f"{cfg.out_file}.pt")

        print(f"Epoch [{epoch + 1}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")

    print("Training complete. Best model saved as 'best_model.pth'.")

