import torch


def get_optimizer_and_lr_schedule(model, optimizer_type,
                                  lr, extractor_lr,
                                  lr_decay, lr_gamma, lr_milestones,
                                  early_stopping_epochs,
                                  weight_decay,
                                  ):
    if extractor_lr != .0:
        params = [
            {
                "params": list(model.parameters())[:-2],
                "lr": extractor_lr
            },
            {
                "params": list(model.parameters())[-2:]
            }
        ]
    else:
        params = model.parameters()

    optimizer = None
    if optimizer_type == 'SGD':
        optimizer = torch.optim.SGD(
            params, lr=lr, weight_decay=weight_decay,
        )
    elif optimizer_type == 'Adam':
        optimizer = torch.optim.Adam(
            params, lr=lr, weight_decay=weight_decay,
        )
    elif optimizer_type == 'RMSprop':
        optimizer = torch.optim.RMSprop(
            params, lr=lr, weight_decay=weight_decay,
        )
    else:
        raise ValueError('unknown optimizer type %s' % optimizer_type)

    if lr_decay == 'step':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            lr_gamma,
        )
    elif lr_decay == 'exponential':
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            lr_gamma,
        )
    elif lr_decay == 'multistep':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            lr_milestones,
            lr_gamma,
        )
    elif lr_decay == 'minloss':
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=lr_gamma,
            patience=early_stopping_epochs - 2,
        )
    else:
        lr_scheduler = None
        # torch.optim.lr_scheduler.LambdaLR(optimizer, )
        # torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, )
        # torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, )
        # torch.optim.lr_scheduler.CyclicLR(optimizer, )
        # torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, )
        # torch.optim.lr_scheduler.OneCycleLR(optimizer, )

    return optimizer, lr_scheduler
