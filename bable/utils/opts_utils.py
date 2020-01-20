import argparse


def parse_args(args=None):
    parser = argparse.ArgumentParser(description='Relative Attributes Scripts')

    # model
    parser.add_argument('--model_type', type=str, default="drn")
    parser.add_argument('--extractor_type', type=str, default="vgg16")

    # dataset
    parser.add_argument('--dataset_type', type=str, default="zappos_v1")
    parser.add_argument('--category_id', type=int, default=0)
    parser.add_argument('--include_equal', action="store_true")
    parser.add_argument('--num_workers', type=int, default=10)

    parser.add_argument('--is_bgr', action="store_true")
    parser.add_argument('--argument_brightness', type=float, default=0.1)
    parser.add_argument('--argument_contrast', type=float, default=0.1)
    parser.add_argument('--argument_saturation', type=float, default=0.1)
    parser.add_argument('--argument_hue', type=float, default=0.1)
    parser.add_argument('--argument_crop_height', type=int, default=224)
    parser.add_argument('--argument_crop_width', type=int, default=224)
    parser.add_argument('--argument_min_scale', type=float, default=0.75)
    parser.add_argument('--argument_max_scale', type=float, default=1.)
    parser.add_argument('--argument_min_ratio', type=float, default=3/4.)
    parser.add_argument('--argument_max_ratio', type=float, default=4/3.)
    parser.add_argument('--val_reisze_height', type=int, default=256)
    parser.add_argument('--val_reisze_width', type=int, default=256)

    # dataloader
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--val_batch_size', type=int, default=16)

    # training
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--extractor_lr', type=float, default=1e-5)
    parser.add_argument('--loss_type', type=str, default="ranknet")
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--optimizer_type', type=str, default="RMSprop")

    # logs
    parser.add_argument('--log_interval_steps', type=int, default=20)

    # gpu
    parser.add_argument('--gpus', type=int, default=0)
    parser.add_argument('--gpu_devices', type=str, default="3")

    return parser.parse_args(args)
