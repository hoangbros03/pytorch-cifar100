#test.py
#!/usr/bin/env python3

import copy
import argparse

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from conf import settings
from utils import get_network, get_test_dataloader

def test_model(model):
    model.eval()

    correct_1 = 0.0
    correct_5 = 0.0

    acc = 0.0
    with torch.no_grad():
        for n_iter, (image, label) in enumerate(cifar100_test_loader):
            print("iteration: {}\ttotal {} iterations".format(n_iter + 1, len(cifar100_test_loader)))

            if args.gpu:
                image = image.cuda()
                label = label.cuda()
                print('GPU INFO.....')
                print(torch.cuda.memory_summary(), end='')


            output = model(image)
            _, pred = output.topk(5, 1, largest=True, sorted=True)

            label = label.view(label.size(0), -1).expand_as(pred)
            correct = pred.eq(label).float()
            acc += correct.sum()

            #compute top 5
            correct_5 += correct[:, :5].sum()

            #compute top1
            correct_1 += correct[:, :1].sum()

    if args.gpu:
        print('GPU INFO.....')
        print(torch.cuda.memory_summary(), end='')

    print()
    print("Top 1 err: ", 1 - correct_1 / len(cifar100_test_loader.dataset))
    print("Top 5 err: ", 1 - correct_5 / len(cifar100_test_loader.dataset))
    print(f"acc: {acc.float() / len(cifar100_test_loader.dataset)}")
    print("Parameter numbers: {}".format(sum(p.numel() for p in model.parameters())))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-weights', type=str, required=True, help='the weights file you want to test')
    parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
    parser.add_argument('-b', type=int, default=16, help='batch size for dataloader')
    args = parser.parse_args()
    
    net = get_network(args)
    model = copy.deepcopy(net)
    
    model_int8 = torch.quantization.quantize_dynamic(
    model,  # the original model
    {torch.nn.Linear, torch.nn.Sequential},  # a set of layers to dynamically quantize
    dtype=torch.qint8)  # the target dtype for quantized weights
    if args.gpu:
        model_int8.to('cuda')
    else:
        model_int8.to('cpu')
        
    # Load state_dict
    if args.gpu:
        model_int8.load_state_dict(torch.load(args.weights, map_location = torch.device('cuda')))
    else:
        model_int8.load_state_dict(torch.load(args.weights, map_location = torch.device('cpu')))
        
    # Load data
    cifar100_test_loader = get_test_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        #settings.CIFAR100_PATH,
        num_workers=4,
        batch_size=args.b,
    )
    
    # Test part
    test_model(model_int8)