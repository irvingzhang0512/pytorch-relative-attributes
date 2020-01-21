import torch
import os
import platform
import shutil
import math
if 'Windows' in platform.platform():
    import sys
    sys.path.append("E:\\vscprojects\\pytorch-relative-attributes")
from tensorboardX import SummaryWriter
from bable.utils.metrics_utils import ScoresAccuracyTool
from bable.utils.opts_utils import parse_args
from bable.builders import datasets_builder, models_builder, loss_builder
from bable.utils.transforms_utils import get_default_transforms_config


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
    val_split = 'val' if 'place_pulse' in args.dataset_type else 'test'
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

    print('dataset init successfully...')
    return train_loader, val_loader


def train_one_epoch(model,
                    train_loader, device,
                    optimizer, loss_fn,
                    epoch, writer, args):
    model.train()
    accuracy_tool = ScoresAccuracyTool()
    base_global_step = (epoch - 1) * \
        math.ceil(len(train_loader.dataset)/args.batch_size)
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

        if idx % args.log_interval_steps == 0:
            print('Train Epoch: {} {}/{} {:.4f} {:.4f}'.format(
                epoch, idx*len(labels), len(train_loader.dataset),
                loss, accuracy_tool.accuracy()
            ))
        if idx % args.summary_interval_steps == 0:
            writer.add_scalars(
                'metrics/accuracy',
                {'train': accuracy_tool.accuracy()},
                base_global_step + idx
            )
            writer.add_scalars(
                'metrics/loss',
                {'train': loss},
                base_global_step + idx
            )
            writer.flush()


def eval(model, val_loader, device, loss_fn, epoch, writer, global_step):
    model.eval()

    total_loss = .0
    accuracy_tool = ScoresAccuracyTool()
    batch_idx = 0

    with torch.no_grad():
        for d in val_loader:
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
    accuracy = accuracy_tool.accuracy()
    print('Val Epoch: {}, average loss {:.4f}, accuracy {:.4f}'.format(
        epoch, total_loss, accuracy
    ))
    writer.add_scalars('metrics/accuracy', {'eval': accuracy}, global_step)
    writer.add_scalars('metrics/loss', {'eval': total_loss}, global_step)
    writer.flush()
    return total_loss, accuracy


def _get_optimizer(model, args):
    # TODO: lr decay
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


def _get_model_dir_name(args, category_name):
    return "logs-{}_{}_{}-{}_{}-loss_{}-{}_{}_{}-wd{}-{}".format(
        args.dataset_type, category_name, args.batch_size,
        args.model_type, args.extractor_type,
        args.loss_type,
        args.optimizer_type, args.lr, args.extractor_lr,
        args.weight_decay,
        args.logs_name,
    )


def train(model, train_loader, val_loader, model_dir, args):
    # remove/create dirs
    eval_dir = os.path.join(model_dir, args.eval_dir_name)
    if args.clean_model_dir and os.path.exists(model_dir):
        shutil.rmtree(model_dir)
    if not os.path.exists(model_dir):
        os.makedirs(eval_dir)

    # summary writer
    writer = SummaryWriter(model_dir)
    writer.add_graph(model, (torch.rand(16, 3, 224, 224),
                             torch.rand(16, 3, 224, 224)))
    writer.flush()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    loss_fn = loss_builder.build_loss(args.loss_type)
    optimizer = _get_optimizer(model, args)
    # for p in model.parameters():
    #     print(p.size())

    min_loss = 1e10
    max_accuracy = .0
    early_stopping_cnt = 0
    for i in range(args.epochs):
        # train
        train_one_epoch(
            model, train_loader, device,
            optimizer, loss_fn,
            i + 1, writer, args,
        )

        # eval
        global_step = (i+1) * \
            math.ceil(len(train_loader.dataset)/args.batch_size)
        cur_loss, cur_accuracy = eval(
            model, val_loader, device, loss_fn, i + 1, writer, global_step
        )

        # save model
        if min_loss > cur_loss:
            file_names = os.listdir(eval_dir)
            prefix = args.min_loss_ckpt_name[:6]
            for file_name in file_names:
                if prefix in file_name:
                    os.remove(os.path.join(eval_dir, file_name))
                    break
            min_loss = cur_loss
            torch.save(model, os.path.join(
                eval_dir,
                args.min_loss_ckpt_name.format(min_loss)
            ))

        if max_accuracy < cur_accuracy:
            file_names = os.listdir(eval_dir)
            prefix = args.max_accuracy_ckpt_name[:6]
            for file_name in file_names:
                if prefix in file_name:
                    os.remove(os.path.join(eval_dir, file_name))
                    break
            max_accuracy = cur_accuracy
            torch.save(model, os.path.join(
                eval_dir,
                args.max_accuracy_ckpt_name.format(max_accuracy)
            ))
            early_stopping_cnt = 0
        else:
            early_stopping_cnt += 1
            if early_stopping_cnt >= args.early_stopping_epochs:
                print('early stopping')
                break

        # save one step model
        torch.save(
            model.state_dict(),
            os.path.join(model_dir, args.step_ckpt_name)
        )
    print('dataset {}/{} finally get loss {:.4f} and accuracy {:.4f}'.format(
        train_loader.dataset.category_name, args.dataset_type,
        min_loss, max_accuracy
    ))
    writer.close()


def main(args):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_devices
    train_loader, val_loader = _get_datasets(args)
    model = models_builder.build_model(
        args.model_type,
        extractor_type=args.extractor_type,
    )
    model_dir = os.path.join(
        args.logs_root_dir,
        _get_model_dir_name(args, train_loader.dataset.category_name)
    )
    train(model, train_loader, val_loader, model_dir, args)


if __name__ == '__main__':
    main(parse_args())
