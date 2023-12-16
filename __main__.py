import argparse
# import os
import pickle
import torch

from .train import execute

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed_init", default=1)
    parser.add_argument("--seed_batch", default=10000)
    parser.add_argument("--seed_trainset", default=-1)
    parser.add_argument("--seed_testset", default=0)

    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--ptr", type=int, required=True)
    parser.add_argument("--pte", type=int, required=True)
    parser.add_argument("--d", type=int)
    parser.add_argument("--num_classes", type=int)

    parser.add_argument("--arch", type=str, required=True)
    parser.add_argument("--act", type=str, default="relu")
    parser.add_argument("--L", type=int)
    parser.add_argument("--h", type=int)

    parser.add_argument("--loss", type=str, default="cross_entropy") 
    parser.add_argument("--dynamics", type=str, default="sgd")
    parser.add_argument("--bs", type=int, required=True)
    parser.add_argument("--dt", type=float, default=0.0)
    parser.add_argument("--dt_eff", type=float, default=0.0)
    parser.add_argument("--momentum", type=float)

    parser.add_argument("--max_wall", type=float, required=True)
    parser.add_argument("--max_step", type=float, default=float('inf'))

    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--dtype", type=str, default="f32")
    args = parser.parse_args().__dict__

    assert args['dt'] + args['dt_eff'], "You need to choose at least one of dt or dt_eff"
    if args['dt_eff'] > 0.0:
        args['dt'] =  args['dt_eff'] * args['h']

    if args['dtype'] == "f64":
        torch.set_default_dtype(torch.float64)
    else:
        torch.set_default_dtype(torch.float32)
        assert args['dtype'] == "f32"

    with open(args['output'], 'wb') as handle:
        pickle.dump(args, handle)


    for data in execute(**args, yield_time=10.0):
        data['args'] = args
        with open(args['output'], 'wb') as handle:
            pickle.dump(args, handle)
            pickle.dump(data, handle)


if __name__ == "__main__":
    main()