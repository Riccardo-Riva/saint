import torch
from torch import nn

import torch.optim as optim
from utils.augmentations import embed_data_mask
from utils.augmentations import add_noise
from utils.augmentations import mixup_data

from utils.gpuusage import print_gpu_utilization, print_summary

import tqdm

def loss_on_batch(
        model,
        data, 
        device,
        pt_tasks = ['contrastive','denoising'],
        pt_aug_dict = {
            'noise_type' : ['mixup','cutmix'],
            'lambda' : 0.1
            },
        pt_contrastive_dict = {
            'pt_projhead_style' : 'diff',
            'nce_temp' : 0.7,
            },
        pt_lam_dict = {
            'lam0' : 0.5,
            'lam1' : 10,
            'lam2' : 1,
            'lam3' : 10
            },
        ):
    
    criterion1 = nn.CrossEntropyLoss()
    criterion2 = nn.MSELoss()
    
    x_categ, x_cont, _ ,cat_mask, con_mask = data[0].to(device), data[1].to(device),data[2].to(device),data[3].to(device),data[4].to(device)
    # embed_data_mask function is used to embed both categorical and continuous data.
    if 'cutmix' in pt_aug_dict['noise_type']:
        x_categ_corr, x_cont_corr = add_noise(x_categ, x_cont, noise_params = pt_aug_dict)
        _ , x_categ_enc_2, x_cont_enc_2 = embed_data_mask(x_categ_corr, x_cont_corr, cat_mask, con_mask,model)
    else:
        _ , x_categ_enc_2, x_cont_enc_2 = embed_data_mask(x_categ, x_cont, cat_mask, con_mask,model)
    _ , x_categ_enc, x_cont_enc = embed_data_mask(x_categ, x_cont, cat_mask, con_mask,model)
    
    if 'mixup' in pt_aug_dict['noise_type']:
        x_categ_enc_2, x_cont_enc_2 = mixup_data(x_categ_enc_2, x_cont_enc_2, device, lam=pt_aug_dict['lambda'])

    loss = 0.0

    if 'contrastive' in pt_aug_dict['noise_type']:
        aug_features_1  = model.transformer(x_categ_enc, x_cont_enc)
        aug_features_2 = model.transformer(x_categ_enc_2, x_cont_enc_2)
        aug_features_1 = (aug_features_1 / aug_features_1.norm(dim=-1, keepdim=True)).flatten(1,2)
        aug_features_2 = (aug_features_2 / aug_features_2.norm(dim=-1, keepdim=True)).flatten(1,2)
        if  pt_contrastive_dict['pt_projhead_style']== 'diff':
            aug_features_1 = model.pt_mlp1(aug_features_1)
            aug_features_2 = model.pt_mlp2(aug_features_2)
        elif pt_contrastive_dict['pt_projhead_style'] == 'same':
            aug_features_1 = model.pt_mlp1(aug_features_1)
            aug_features_2 = model.pt_mlp1(aug_features_2)
        else:
            print('Not using projection head')
        logits_per_aug1 = aug_features_1 @ aug_features_2.t()/pt_contrastive_dict['nce_temp']
        logits_per_aug2 = aug_features_2 @ aug_features_1.t()/pt_contrastive_dict['nce_temp']
        targets = torch.arange(logits_per_aug1.size(0)).to(logits_per_aug1.device)
        loss_1 = criterion1(logits_per_aug1, targets)
        loss_2 = criterion1(logits_per_aug2, targets)
        loss   = pt_lam_dict['lam0']*(loss_1 + loss_2)/2
        
    elif 'contrastive_sim' in pt_tasks:
        aug_features_1  = model.transformer(x_categ_enc, x_cont_enc)
        aug_features_2 = model.transformer(x_categ_enc_2, x_cont_enc_2)
        aug_features_1 = (aug_features_1 / aug_features_1.norm(dim=-1, keepdim=True)).flatten(1,2)
        aug_features_2 = (aug_features_2 / aug_features_2.norm(dim=-1, keepdim=True)).flatten(1,2)
        aug_features_1 = model.pt_mlp1(aug_features_1)
        aug_features_2 = model.pt_mlp2(aug_features_2)
        c1 = aug_features_1 @ aug_features_2.t()
        loss+=pt_lam_dict['lam1']*torch.diagonal(-1*c1).add_(1).pow_(2).sum()

    if 'denoising' in pt_tasks:
        cat_outs, con_outs = model(x_categ_enc_2, x_cont_enc_2)
        # if con_outs.shape(-1) != 0:
        # import ipdb; ipdb.set_trace()
        if len(con_outs) > 0:
            con_outs =  torch.cat(con_outs,dim=1)
            l2 = criterion2(con_outs, x_cont)
        else:
            l2 = 0
        l1 = 0
        # import ipdb; ipdb.set_trace()
        n_cat = x_categ.shape[-1]
        for j in range(1,n_cat):
            l1+= criterion1(cat_outs[j],x_categ[:,j])

        loss += pt_lam_dict['lam2']*l1 + pt_lam_dict['lam3']*l2

    return loss


def epoch_train(
        model,
        loader,
        optimizer,
        device,
        pt_tasks = ['contrastive','denoising'],
        pt_aug_dict = {
            'noise_type' : ['mixup','cutmix'],
            'lambda' : 0.1
            },
        pt_contrastive_dict = {
            'pt_projhead_style' : 'diff',
            'nce_temp' : 0.7,
            },
        pt_lam_dict = {
            'lam0' : 0.5,
            'lam1' : 10,
            'lam2' : 1,
            'lam3' : 10
            },
        ):

    model.train()

    running_loss = 0.0

    for i, data in enumerate(loader, 0):
        optimizer.zero_grad()

        loss = loss_on_batch(
            model,
            data,
            device,
            pt_tasks=pt_tasks,
            pt_aug_dict=pt_aug_dict,
            pt_contrastive_dict=pt_contrastive_dict,
            pt_lam_dict=pt_lam_dict
        )

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if i+1 % 2**8 == 0:    
            print(f'Batch {i} of {len(loader)} processed')

    running_loss /= len(loader)

    return running_loss


def epoch_eval(
        model,
        loader,
        device,
        pt_tasks = ['contrastive','denoising'],
        pt_aug_dict = {
            'noise_type' : ['mixup','cutmix'],
            'lambda' : 0.1
            },
        pt_contrastive_dict = {
            'pt_projhead_style' : 'diff',
            'nce_temp' : 0.7,
            },
        pt_lam_dict = {
            'lam0' : 0.5,
            'lam1' : 10,
            'lam2' : 1,
            'lam3' : 10
            },
        ):

    running_loss = 0.0

    model.eval()

    with torch.no_grad():
        for i, data in enumerate(loader, 0):
            
            loss = loss_on_batch(
                model,
                data,
                device,
                pt_tasks=pt_tasks,
                pt_aug_dict=pt_aug_dict,
                pt_contrastive_dict=pt_contrastive_dict,
                pt_lam_dict=pt_lam_dict
            )

            running_loss += loss.item()

            if i+1 % 2**8 == 0:    
                print(f'Batch {i} of {len(loader)} processed')

    running_loss /= len(loader)

    return running_loss
    

########## PRETRAINING PIPELINE #############
def pretrain(model,trainloader,valloader,opt,device):

    optimizer = optim.AdamW(model.parameters(),lr=opt.lr)
    
    pt_tasks = opt.pt_tasks
    pt_aug_dict = {
        'noise_type' : opt.pt_aug,
        'lambda' : opt.pt_aug_lam
    }
    pt_contrastive_dict = {
        'pt_projhead_style' : opt.pt_projhead_style,
        'nce_temp' : opt.nce_temp,
    }
    pt_lam_dict = {
        'lam0' : opt.lam0,
        'lam1' : opt.lam1,
        'lam2' : opt.lam2,
        'lam3' : opt.lam3
    }

    train_losses = []
    val_losses = []

    print('\n')
    print(f'Number of trainloader batches: {len(trainloader)}')
    print(f'Number of valloader batches: {len(valloader)}')

    print("First evaluation on the validation set")
    best_val_loss = epoch_eval(
        model,
        valloader,
        device,
        pt_tasks=opt.pt_tasks,
        pt_aug_dict=pt_aug_dict,
        pt_contrastive_dict=pt_contrastive_dict,
        pt_lam_dict=pt_lam_dict
    )
    bestmodel_state_dict = model.state_dict()
    print(f'Validation Loss: {best_val_loss}')


    print("Pretraining begins!")
    for epoch in tqdm.tqdm(range(opt.pretrain_epochs)):

        print(f'\nPretraining epoch: {epoch}')
        train_running_loss = epoch_train(
            model,
            trainloader,
            optimizer,
            device,
            pt_tasks=opt.pt_tasks,
            pt_aug_dict=pt_aug_dict,
            pt_contrastive_dict=pt_contrastive_dict,
            pt_lam_dict=pt_lam_dict
        )
        #print(f"Memory allocated: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")
        #print(f"Max memory allocated: {torch.cuda.max_memory_allocated() / 1024 ** 3:.2f} GB")
        if epoch == 0:
            memory_used, _ = print_gpu_utilization(device)

        print(f'Epoch: {epoch}, Training running Loss: {train_running_loss}')

        train_loss = epoch_eval(
            model,
            trainloader,
            device,
            pt_tasks=opt.pt_tasks,
            pt_aug_dict=pt_aug_dict,
            pt_contrastive_dict=pt_contrastive_dict,
            pt_lam_dict=pt_lam_dict
        )
        print(f'Epoch: {epoch}, Traning Loss: {train_loss}')

        val_loss = epoch_eval(
            model,
            valloader,
            device,
            pt_tasks=opt.pt_tasks,
            pt_aug_dict=pt_aug_dict,
            pt_contrastive_dict=pt_contrastive_dict,
            pt_lam_dict=pt_lam_dict
        )
        print(f'Epoch: {epoch}, Validation Loss: {val_loss}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            bestmodel_state_dict = model.state_dict()

        train_losses.append(train_loss)
        val_losses.append(val_loss)

    print('END OF PRETRAINING!')
    return model, bestmodel_state_dict, train_losses, val_losses, memory_used
