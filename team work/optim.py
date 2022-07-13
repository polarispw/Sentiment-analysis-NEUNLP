from transformers import optimization as optim


def collate_para(model, base_lr, lr_decay, discr=False):
    no_decay = ['bias', 'gamma', 'beta']
    if discr:
        if len(base_lr) > 1:
            groups = [(f'layer.{i}.', base_lr[i]) for i in range(12)]
        else:
            lr = base_lr[0]
            groups = [(f'layer.{i}.', lr * pow(lr_decay, 11 - i)) for i in range(12)]
        group_all = [f'layer.{i}.' for i in range(12)]
        no_decay_optimizer_parameters = []
        decay_optimizer_parameters = []
        for g, l in groups:
            decay_optimizer_parameters.append(
                {
                    'params': [p for n, p in model.named_parameters() if
                               not any(nd in n for nd in no_decay) and any(nd in n for nd in [g])],
                    'weight_decay_rate': 0.01, 'lr': l
                }
            )
            no_decay_optimizer_parameters.append(
                {
                    'params': [p for n, p in model.named_parameters() if
                               any(nd in n for nd in no_decay) and any(nd in n for nd in [g])],
                    'weight_decay_rate': 0.0, 'lr': l
                }
            )

        group_all_parameters = [
            {'params': [p for n, p in model.named_parameters() if
                        not any(nd in n for nd in no_decay) and not any(nd in n for nd in group_all)],
             'weight_decay_rate': 0.01},
            {'params': [p for n, p in model.named_parameters() if
                        any(nd in n for nd in no_decay) and not any(nd in n for nd in group_all)],
             'weight_decay_rate': 0.0},
        ]
        optimizer_parameters = no_decay_optimizer_parameters + decay_optimizer_parameters + group_all_parameters

    else:
        optimizer_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.01},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.0}
        ]
    return optimizer_parameters


def build_scheduler(scheduler_name, optimizer, wp_steps, train_steps=0):
    sch_list = ["constant", "linear", "cosine"]
    assert scheduler_name in sch_list, "Choose scheduler from [\"constant\", \"linear\", \"cosine\"]."
    if scheduler_name == sch_list[0]:
        return optim.get_constant_schedule(optimizer, wp_steps)
    if scheduler_name == sch_list[1]:
        return optim.get_linear_schedule_with_warmup(optimizer, wp_steps, train_steps)
    if scheduler_name == sch_list[2]:
        return optim.get_cosine_schedule_with_warmup(optimizer, wp_steps, train_steps)
