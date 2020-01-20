import os
import torch
from tqdm import tqdm
from bable.utils.transforms_utils import get_default_transforms_config
from bable.builders import datasets_builder, models_builder, loss_builder
from bable.utils.opts_utils import parse_args
from bable.utils.metrics_utils import ScoresAccuracyTool


def _get_datasets(args):
    train_dataset_config = get_default_transforms_config()
    train_dataset_config['is_rgb'] = not args.is_bgr
    train_dataset_config['brightness'] = args.argument_brightness
    train_dataset_config['contrast'] = args.argument_contrast
    train_dataset_config['saturation'] = args.argument_saturation
    train_dataset_config['hue'] = args.argument_hue
    train_dataset_config['random_resized_crop_size'] = (
        args.argument_crop_height, args.argument_crop_width
    )
    train_dataset_config['random_resized_crop_scale'] = (
        args.argument_min_scale, args.argument_max_scale
    )
    train_dataset_config['random_resized_crop_ratio'] = (
        args.argument_min_ratio, args.argument_max_ratio
    )
    train_dataset = datasets_builder.build_dataset(
        dataset_type=args.dataset_type,
        split='train',
        category_id=args.category_id,
        trans_config=train_dataset_config,
        include_equal=args.include_equal,
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    val_dataset_config = get_default_transforms_config(False)
    val_dataset_config['resize_size'] = (
        args.val_reisze_height, args.val_reisze_width
    )
    val_split = 'test' if 'zappos' in args.dataset_type else 'val'
    val_dataset = datasets_builder.build_dataset(
        dataset_type=args.dataset_type,
        split=val_split,
        category_id=args.category_id,
        trans_config=val_dataset_config,
        include_equal=args.include_equal,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.val_batch_size,
        num_workers=args.num_workers,
    )

    return train_loader, val_loader


def train_one_epoch(model,
                    train_loader, device,
                    optimizer, loss_fn,
                    epoch,
                    log_interval_steps):
    model.train()
    accuracy_tool = ScoresAccuracyTool()
    for idx, d in enumerate(train_loader):
        img1 = d[0][0].to(device)
        img2 = d[0][1].to(device)
        labels = d[1].to(device)
        optimizer.zero_grad()
        s1_logits, s2_logits = model(img1, img2)
        accuracy_tool.update(
            s1_logits.cpu().detach().numpy(),
            s2_logits.cpu().detach().numpy(),
            labels.cpu().numpy()
        )
        loss = loss_fn(s1_logits, s2_logits, labels)
        loss.backward()
        optimizer.step()

        if idx % log_interval_steps == 0:
            print('Train Epoch: {} {}/{} {:.4f} {:.2f}'.format(
                epoch, idx*len(labels), len(train_loader.dataset),
                loss, accuracy_tool.accuracy()
            ))


def eval(model, val_loader, device, loss_fn, epoch, args):
    model.eval()

    total_loss = .0
    accuracy_tool = ScoresAccuracyTool()
    batch_idx = 0

    with torch.no_grad():
        for d in tqdm(val_loader):
            batch_idx += 1
            img1 = d[0][0].to(device)
            img2 = d[0][1].to(device)
            labels = d[1].to(device)
            s1_logits, s2_logits = model(img1, img2)
            total_loss += loss_fn(s1_logits, s2_logits, labels)
            accuracy_tool.update(
                s1_logits.cpu().numpy(),
                s2_logits.cpu().numpy(),
                labels.cpu().numpy()
            )

    total_loss = total_loss / batch_idx
    print('Val Epoch: {}, average loss {:.4f}, accuracy {:.4f}'.format(
        epoch, total_loss, accuracy_tool.accuracy()
    ))


def _get_optimizer(model, args):
    def _get_lr():
        return args.lr

    if args.extractor_lr != .0:
        params = [
            {
                "params": list(model.parameters())[:-2],
                "lr": args.extractor_lr
            },
            {
                "params": list(model.parameters())[-2:]
            }
        ]
    else:
        params = model.parameters()
    if args.optimizer_type == 'SGD':
        return torch.optim.SGD(
            params, lr=_get_lr(), weight_decay=args.weight_decay,
        )
    elif args.optimizer_type == 'Adam':
        return torch.optim.Adam(
            params, lr=_get_lr(), weight_decay=args.weight_decay,
        )
    elif args.optimizer_type == 'RMSprop':
        return torch.optim.RMSprop(
            params, lr=_get_lr(), weight_decay=args.weight_decay,
        )
    raise ValueError('unknown optimizer type %s' % args.optimizer_type)


def train(model, train_loader, val_loader, args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    loss_fn = loss_builder.build_loss(args.loss_type)
    optimizer = _get_optimizer(model, args)
    for p in model.parameters():
        print(p.size())

    for i in range(args.epochs):
        train_one_epoch(
            model, train_loader, device,
            optimizer, loss_fn,
            i + 1, args.log_interval_steps
        )
        eval(model, val_loader, device, loss_fn, i + 1, args)

        # TODO: save model
        # save max accuracy/min loss model
        # save one step model
        # torch.save(model, "siamese{}.pt".format(i))


def main(args):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_devices
    train_loader, val_loader = _get_datasets(args)
    model = models_builder.build_model(
        args.model_type,
        extractor_type=args.extractor_type,
    )
    train(model, train_loader, val_loader, args)


if __name__ == '__main__':
    main(parse_args())
