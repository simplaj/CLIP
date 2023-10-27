import torch
from torch.utils.data import random_split

from parse import args
from data import Angle


def train(model, dataloader, testloader, cfg):
    pname = cfg.pname
    device = cfg.device
    lr = cfg.lr
    epochs = cfg.epochs
    warmup_epochs = cfg.warmup_epochs
    weight_decay = cfg.wd
    
    wandb.init(
        # set the wandb project where this run will be logged
        project=pname,
        
        # track hyperparameters and run metadata
        config=cfg
    )
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = WarmupCosineLR(optimizer, warmup_epochs, epochs, 1, -1)
    for epoch in range(epochs):
        model.train()
        acc = []
        recall = []
        loss_all = []
        for b, data in enumerate(dataloader):
            im, label = data['data'], data['label']
            im = im.to(device)
            label = label.to(device)
            preds = model(im)
            
            loss = criterion(preds, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            acc_, recall_ = cal_acc_recall(preds, label, 3)
            if b % 10 == 0:
                print(f'epoch:{epoch} loss:{loss.item():.3f} acc:{acc_:.3f}', end=' ')  # {recall_[2]:.3f} {recall_[3]:.3f}')
                [print(f'{x:.3f}', end=' ') for x in recall_]
                print()
                acc.append(acc_)
            recall.append(sum(recall_) / len(recall_))
            loss_all.append(loss)
            
        eval_acc = gp_evaluate(model, testloader, cfg)
        torch.save(model, cfg.path + f'/GPR_{epoch}.pth')
        scheduler.step()
        train_lr = optimizer.param_groups[0]['lr']
        status = {'train_loss': sum(loss_all) / len(loss_all),
                'train_acc':sum(acc) / len(acc),
                'eval_acc': eval_acc,
                'train_recall':sum(recall) / len(recall),
                'lr':train_lr}
        [print(f'{k} {v:.3f}', end=' ') for k, v in status.items()]
        print()
        wandb.log(status)
        

class M1027(nn.Module):
    """Some Information about GPR"""
    def __init__(self, num_classes):
        super(GPR, self).__init__()
        self.features = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT).features
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(1280, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        # Cannot use "squeeze" as batch-size can be 1
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

        
if __name__ == '__main__':
    cfg = args
    model = M1027(3)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    if cfg.mode == 'train':
        dataset = Angle(mode='train', preprocess=preprocess)
        train(model, dataloader, testloader, cfg)