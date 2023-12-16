# import threading
import time
# from functools import partial
from itertools import count
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .architecture import Linear, mlp, deepLinear, randomFeatures, cnn, ViT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def init(arch, h, act, seed_init, **args):
    """initialize weights and biases
        arch (str): architecture
        h (int): hidden dimension
        act (str): activation function
        seed_init (int): seed for initialization
        args (dict): additional arguments
    """

    if act == 'relu':
        act = nn.ReLU()
    if act == 'silu':
        act = nn.SiLU()
    if act == 'gelu':
        act = nn.GELU()
    if act == 'softplus':
        act = nn.Softplus()

    xtr, ytr, xte, yte = dataset(**args)
    xtr, ytr, xte, yte = xtr.to(device), ytr.to(device), xte.to(device), yte.to(device)
    num_classes = torch.cat((ytr, yte)).max().item() + 1
    input_dim = xtr.view(xtr.size(0), -1).shape[1]

    print(f'dataset generated xtr.shape={xtr.shape} xte.shape={xte.shape}', flush=True)
    print(f'dataset on device {xtr.device}', flush=True)

    if arch == 'linear':
        model = Linear(input_dim, num_classes).to(device)

    if arch == 'deepLinear':
        model = deepLinear(input_dim, h, args.get("L"), num_classes).to(device)

    if arch == 'randomFeatures':
        model = randomFeatures(input_dim, h, args.get("L"), num_classes, act).to(device)

    if arch == 'mlp':
        model = mlp(input_dim, h, args.get("L"), num_classes, act).to(device) 

    if arch == 'cnn':
        assert len(xtr.shape) == 4 # cnn input must have shape (N, C, H, W)
        in_dim = xtr.shape[2]*xtr.shape[3]
        in_channels = xtr.shape[1]
        model = cnn(in_dim, in_channels, out_channels=h, num_classes=num_classes, act=act).to(device)

    if arch == 'vit': 
        assert len(xtr.shape) == 4 # Vision Transformer input must have shape (N, C, H, W)
        in_dim = xtr.shape[2]*xtr.shape[3]
        in_channels = xtr.shape[1]
        patch_size = 4
        num_patches = in_dim // patch_size**2
        model = ViT(embed_dim=h, hidden_dim=2*h, num_channels=in_channels, num_heads=1, num_layers=1, num_classes=num_classes, patch_size=patch_size, num_patches=num_patches, dropout=0.0).to(device)

    torch.manual_seed(seed_init)
    def init_weights(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            torch.nn.init.normal_(m.weight, mean=0.0, std=1.0)
            if m.bias is not None:
                torch.nn.init.normal_(m.bias, mean=0.0, std=1.0)
    model.apply(init_weights)

    print('network initialized', flush=True)

    return model, xtr, ytr, xte, yte


def init_loss(loss):
    if loss == "hinge":
        return lambda f,y: nn.MultiMarginLoss(reduction='none')(f,y)
    elif loss == "cross_entropy":
        return lambda f,y: F.cross_entropy(f, y)
    else:
        raise ValueError("{loss} is not a valid choice of loss")
    

def init_optimizer(dynamics, model, dt):
    '''initialize optimizer
        dynamics (str): optimizer
        model (nn.Module): model
        dt (float): learning rate
        args (dict): additional arguments

    Returns:
        optimizer (torch.optim): optimizer
    '''
    if dynamics == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=dt)
    elif dynamics == 'adam':
        return torch.optim.Adam(model.parameters(), lr=dt,)
    elif dynamics == 'adamW':
        return torch.optim.AdamW(model.parameters(), lr=dt,)
    else:
        raise ValueError(f"{dynamics} is not a valid choice of dynamics")



def execute(yield_time=0.0, **args):
    print(f"device={device} dtype={torch.ones(3).dtype}", flush=True)

    yield {
        'finished': False,
    }

    model, xtr, ytr, xte, yte = init(**args)
    # train_dl = DataLoader(train_data, batch_size=args['bs'], shuffle=True)
    # test_dl  = DataLoader(test_data,  batch_size=args['bs'])

    darch = dict(
        params_shapes=[p.shape for p in model.parameters()],
        params_sizes=[p.size for p in model.parameters()],
    )

    optimizer = init_optimizer(args['dynamics'], model, dt=args['dt'],)

    loss = init_loss(args['loss'])

    def stop_criterion(train_loss, delta_train_err, step, t, wall_time, check_train_err):
        if not torch.isfinite(train_loss): return True
        if train_loss<=0: 
            print(f"train_loss {train_loss} <= 0", flush=True)
            return True
        if check_train_err and delta_train_err==0: 
            print(f"train_error didn't change in the last epoch: delta train error = {delta_train_err}", flush=True)
            return True
        if step >= args['max_step']: 
            print(f"step {step} >= max_step {args['max_step']}", flush=True)
            return True
        if wall_time >= args['max_wall']: 
            print(f"wall_time {wall_time} >= max_wall {args['max_wall']}", flush=True)
            return True
        return False
    
    def checkpoint_criterion(train_loss, delta_train_err, epoch, step, t, wall_time, check_train_err):
        if stop_criterion(train_loss, delta_train_err, step, t, wall_time, check_train_err): return True
        # if step % 100 == 0: return True
        if step == 0: return True
        if math.log2(step)%1==0.0: return True

    for _, d in train(optimizer=optimizer, model=model, xtr=xtr, ytr=ytr, xte=xte, yte=yte, dt=args['dt'], bs=args['bs'], loss=loss, seed_batch=args['seed_batch'], 
                      checkpoint_criterion= checkpoint_criterion, stop_criterion = stop_criterion,
                      ):

            yield {
                'arch': darch,
                args['dynamics']: dict(dynamics=d),
                'finished': False,
            }

    yield {
        'arch': darch,
        args['dynamics']: dict(dynamics=d),
        'finished': True,
    }


def dataset(dataset, seed_trainset, seed_testset, ptr, pte, **args):
    from .dataset import (cifar10, mnist, gaussian_multiclass)

    if dataset in ['gaussian_multiclass']:
        d = args['d']
        nc = args['num_classes']

        if dataset == 'gaussian_multiclass':
            return gaussian_multiclass(d, nc, seed_trainset, seed_testset, ptr, pte)

    if dataset == 'mnist':
        return mnist(seed_trainset, seed_testset, ptr, pte)

    if dataset == 'cifar10':
        return cifar10(seed_trainset, seed_testset, ptr, pte)

    raise f"{dataset} not available. available are cifar10, mnist, gaussian_multiclass"



def train(
    optimizer, model, xtr, ytr, xte, yte, dt, bs, loss, seed_batch,
    checkpoint_criterion, stop_criterion,
    ckpt_save_parameters=False, ckpt_save_pred=False, **args
):

    wall0 = time.perf_counter()

    def evaluate(f, x, y):
        f.eval()
        with torch.no_grad():
            pred = f(x)
            loss_val = (loss(pred, y)).mean()
            err = (pred.argmax(axis=1) != y).sum() / len(y)
        return pred, loss_val, err

    dynamics = []

    t = 0
    step = 0
    epoch = 0
    epoch_check = 0.0
    train_check = 0.0
    delta_train_err = 1.0
    check_train_err = False
    delta_epoch_check = 10
    gen = torch.random.manual_seed(seed_batch)

    print("starting training", flush=True)

    for iter_index in count():
        start = (iter_index == 0)

        if not start:
            while True:
                model.train()
                idx = torch.randperm(len(ytr), generator=gen)[:bs]
                x_batch, y_batch = xtr[idx], ytr[idx]
                
                pred_batch = model(x_batch)
                loss_batch = loss(pred_batch, y_batch).mean()

                optimizer.zero_grad()
                loss_batch.backward()
                optimizer.step()

                t += dt
                step += 1
                epoch = step * bs / len(ytr)
                      
                train_pred, train_loss, train_err = evaluate(model, xtr, ytr)

                check_train_err = False
                if epoch >= epoch_check+delta_epoch_check:
                    epoch_check += delta_epoch_check
                    delta_train_err = abs(train_err - train_check)
                    check_train_err = True

                wall_time = time.perf_counter() - wall0
                if checkpoint_criterion(train_loss, delta_train_err, epoch, step, t, wall_time, check_train_err): break
                if check_train_err: train_check = train_err

        else:
            train_pred, train_loss, train_err = evaluate(model, xtr, ytr)

        stop = False

        wall_time = time.perf_counter() - wall0
        if stop_criterion(train_loss, delta_train_err, step, t, wall_time, check_train_err): stop = True

        if start:
            print("create state (train)", flush=True)

        train = dict(
            loss=train_loss,
            err=train_err,
            pred  = (train_pred if stop or ckpt_save_pred else None),
            label = ytr if stop or ckpt_save_pred else None,
        )

        if start:
            print("create state (test)", flush=True)

        test_pred, test_loss, test_err = evaluate(model, xte, yte)

        test = dict(
            loss=test_loss,
            err=test_err,
            pred  =(test_pred if stop or ckpt_save_pred else None),
            label = yte if stop or ckpt_save_pred else None,
        )

        if start:
            print("create state", flush=True)

        state = dict(
            t=t,
            step=step,
            epoch=epoch,
            wall=time.perf_counter() - wall0,
            train=train,
            test=test,
            parameters=(p for p in model.paramters()) if (ckpt_save_parameters and (start or stop)) else None,
        )
        dynamics.append(state)

        internal = dict(
            model=model,
            train=dict(
                x=xtr,
                y=ytr,
            ),
            test=dict(
                x=xte,
                y=yte,
            ),
        )

        yield internal, dynamics

        print((
            f"[{step} t={t:.2e} wall={state['wall']:.0f} s={len(dynamics)}] "
            f"[train loss={state['train']['loss']:.2e} err={state['train']['err']:.2f}] "
            f"[test loss={state['test']['loss']:.2e} err={state['test']['err']:.2f}]"
        ), flush=True)

        del state

        if stop:
            return