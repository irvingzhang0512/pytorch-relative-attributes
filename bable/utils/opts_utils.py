import argparse


def parse_args(args=None):
    parser = argparse.ArgumentParser(description='Relative Attributes Scripts')

    # model
    parser.add_argument('--model-type', type=str, default="drn")
    parser.add_argument('--extractor-type', type=str, default="vgg16")

    # dataset
    parser.add_argument('--dataset-type', type=str, default="zappos_v1")
    parser.add_argument('--category-id', type=int, default=0)
    parser.add_argument('--include-equal', action="store_true")
    parser.add_argument('--num-workers', type=int, default=10)

    parser.add_argument('--is-bgr', action="store_true")
    parser.add_argument('--argument-brightness', type=float, default=0.1)
    parser.add_argument('--argument-contrast', type=float, default=0.1)
    parser.add_argument('--argument-saturation', type=float, default=0.1)
    parser.add_argument('--argument-hue', type=float, default=0.1)
    parser.add_argument('--argument-crop-height', type=int, default=224)
    parser.add_argument('--argument-crop-width', type=int, default=224)
    parser.add_argument('--argument-min-scale', type=float, default=0.75)
    parser.add_argument('--argument-max-scale', type=float, default=1.)
    parser.add_argument('--argument-min-ratio', type=float, default=3/4.)
    parser.add_argument('--argument-max-ratio', type=float, default=4/3.)
    parser.add_argument('--val-reisze-height', type=int, default=256)
    parser.add_argument('--val-reisze-width', type=int, default=256)

    # dataloader
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--val-batch-size', type=int, default=16)

    # training
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--extractor-lr', type=float, default=1e-5)
    parser.add_argument('--loss-type', type=str, default="ranknet")
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    parser.add_argument('--optimizer-type', type=str, default="RMSprop")
    parser.add_argument('--early-stopping-epochs', type=int, default=5)

    # logs
    parser.add_argument('--log-interval-steps', type=int, default=20)
    parser.add_argument('--summary-interval-steps', type=int, default=20)
    parser.add_argument('--logs-root-dir', type=str, default="./logs")
    parser.add_argument('--logs-name', type=str, default="default")
    parser.add_argument('--clean-model-dir', action="store_true")
    parser.add_argument('--eval-dir-name', type=str, default="eval")
    parser.add_argument('--step-ckpt-name',
                        type=str, default="step.pth")
    parser.add_argument('--min-loss-ckpt-name',
                        type=str, default="min_loss_{:.4f}.pth")
    parser.add_argument('--max-accuracy-ckpt-name',
                        type=str, default="max_accuracy_{:.4f}.pth")

    # gpu
    parser.add_argument('--gpus', type=int, default=0)
    parser.add_argument('--gpu-devices', type=str, default="3")

    return parser.parse_args(args)
