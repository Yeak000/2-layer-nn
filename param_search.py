from train import *


def param_search(lr_l, hidden_size_l, l2_penalty_l):
    data = load_mnist_data(mnist_dir)
    res = {}
    for lr in tqdm(lr_l):
        for hidden_size in hidden_size_l:
            for l2_penalty in l2_penalty_l:
                _, va, _, _, params = train(data,lr,hidden_size,l2_penalty)
                path = 'lr:'+str(lr)+'  h_s:'+str(hidden_size)+'  l2_penalty:'+str(l2_penalty)
                res[path] = va[-1]
                save_model(params, 'ckpt/'+path+'.pkl')
    return res