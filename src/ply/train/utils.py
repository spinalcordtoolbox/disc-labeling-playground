
def adjust_learning_rate(optimizer, lr, gamma):
    """
    Sets the learning rate to the initial LR decayed by schedule
    Copied from https://github.com/spinalcordtoolbox/disc-labeling-hourglass
    """
    lr *= gamma
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr