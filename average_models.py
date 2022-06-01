import copy
import torch

def create_mean_model_parameters(models:list):
    if len(models)==1:
        return models[0]
    else:
        ########IMPORTANT#################
        #use best model for optimizer settings
        new_model = copy.deepcopy(models[0])
        parameter_keys = list(new_model.get_parameters()["policy"].keys())
        new_parameters = new_model.get_parameters()
        for k in parameter_keys:
            p_list = [m.get_parameters()["policy"][k] for m in models]
            new_parameters["policy"][k] = torch.mean(torch.stack(p_list,dim=-1),dim=-1)
            # print("stop")
        new_model.set_parameters(new_parameters)
        return new_model

def create_weighted_mean_model_parameters(models:list,weights:list):
    if len(models)==1:
        return models[0]
    else:
        ########IMPORTANT#################
        #use best model for optimizer settings
        _,idx = torch.topk(torch.Tensor(weights),1)
        new_model = copy.deepcopy(models[idx.item()])
        parameter_keys = list(new_model.get_parameters()["policy"].keys())
        new_parameters = new_model.get_parameters()
        for k in parameter_keys:
            p_list = [m.get_parameters()["policy"][k] for m in models]
            p_list_weighted = [w*p for w,p in zip(weights,p_list)]
            new_parameters["policy"][k] = torch.sum(torch.stack(p_list_weighted,dim=-1),dim=-1)
            # print("stop")
        new_model.set_parameters(new_parameters)
        #test to check that models actually are different
        #test = [[torch.sum(torch.abs(models[i].get_parameters()["policy"][p])).item() for i in range(5)] for p in parameter_keys]
        return new_model

def create_median_model_parameters(models:list):
    if len(models)==1:
        return models[0]
    else:
        ########IMPORTANT#################
        #use best model for optimizer settings
        new_model = copy.deepcopy(models[0])
        parameter_keys = list(new_model.get_parameters()["policy"].keys())
        new_parameters = new_model.get_parameters()
        for k in parameter_keys:
            p_list = [m.get_parameters()["policy"][k] for m in models]
            new_parameters["policy"][k] = torch.median(torch.stack(p_list,dim=-1),dim=-1)[0]
            # print("stop")
        new_model.set_parameters(new_parameters)
        return new_model






def create_top_n_mean_model_parameters(models:list,performance,n):
    performance = torch.Tensor(performance)
    _,idx = torch.topk(performance,n)
    new_model_list = [models[i] for i in list(idx)]
    model_average = create_mean_model_parameters(new_model_list)
    return model_average

def create_softmax_model_parameters(models:list,performance):
    performance = torch.Tensor(performance)
    weights = torch.nn.functional.softmax(performance)
    print(weights)
    model_average = create_weighted_mean_model_parameters(models,weights)
    return model_average

def create_top_model(models:list,performance:list):
    if len(models)<2:
        raise NotImplementedError
    performance = torch.Tensor(performance)
    _,idx = torch.topk(performance,1)
    new_model = models[idx.item()]
    return new_model

def create_top_n_median_model_parameters(models:list,performance,n):
    performance = torch.Tensor(performance)
    _,idx = torch.topk(performance,n)
    new_model_list = [models[i] for i in list(idx)]
    model_average = create_median_model_parameters(new_model_list)
    return model_average