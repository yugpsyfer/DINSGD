import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.metrics import accuracy_score
from dinsgd import DINSGD
import wandb
from torch.distributed import ReduceOp
from resnet import resnet
from torch.optim import SGD, Adam


def allocate_process_to_GPU(local_rank, args):
     
    world_rank =  local_rank % args['world_size']
    compression_method = args["compression_method"]

    if world_rank == 0:
        wandb.init(
          project=args["project_name"],                             #Project name
          entity=args["entity_name"],                               #Entity Name
          name=args["name"],                                    #Current run name
          config=args['optim_params']                            #Run configurations
      ) 
    
    dist.init_process_group(backend=args["backend"],
    init_method=args["init_method"],
    group_name=args["name"],
    rank=world_rank,
    world_size=args["world_size"])

    torch.cuda.set_device(world_rank)
  
    trainLoader = args['trainLoader']
    testLoader = args['testLoader']


    model, criterion = prepare_model(world_rank)
    model.to(device='cuda')
    
    if args['optim_params']['optim'] == "DINSGD":
        optimizer = DINSGD(model.parameters(),
        p=args['optim_params']['p'], 
        q=args['optim_params']['q'], 
        beta=args['optim_params']['beta'],
        rho=args["world_size"], 
        compression_method=compression_method)

    elif args['optim_params']['optim'] == "SGD":
        optimizer = SGD(model.parameters(),lr=args['optim_params']['lr'],
         weight_decay=args['optim_params']['weight_decay'], 
         momentum=args['optim_params']['momentum'])
    
    elif args['optim_params']['optim'] == "ADAM":
        optimizer = Adam(model.parameters(),
         lr=args['optim_params']['lr'],
         weight_decay=args['optim_params']['weight_decay'], 
         eps=args['optim_params']['eps'])
    
    for epoch in range(args["epochs"]):
        l = train_single_epoch(model, trainLoader, criterion, optimizer)

        if (epoch)%5 == 0: 
            model = average_models(model, world_size=args["world_size"])
            
            if world_rank == 0:
                accu_train = evaluate(model=model, testLoader=trainLoader)
                accu_test = evaluate(model=model, testLoader=testLoader)
                
                update_wandb_logs(loss=l,  
                accu_train=accu_train, 
                accu_test=accu_test,
                epoch=epoch)

                dist.barrier()
            else:
                dist.barrier()


def update_wandb_logs(loss, accu_train, accu_test, epoch):    
    wandb.log({"Training Accuracy": accu_train, "Test Accuracy": accu_test, "Loss":loss}) #Comment this for w/o wandb run
    
    display_string = "Train Accuracy: {:.4f} | Test Accuracy: {:.4f} | Train Loss: {:.4f} | Epoch: {}".format(accu_train, 
    accu_test,
     loss,
     epoch)
    
    print(display_string)


def average_models(model, world_size):

    for module in model.parameters():
        communication_tensor = module.data.clone()
        dist.all_reduce(communication_tensor, op=ReduceOp.SUM)
        communication_tensor.div_(world_size)
        module.data = communication_tensor.clone()    

    return model

    
def train_single_epoch(model, trainLoader, criterion, optimizer):
    batch_count = 0
    total_loss = 0

    for batch in trainLoader: 
        X, y = batch
        X = X.to('cuda')
        y = y.to('cuda')
        model.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        total_loss+=loss.detach()
        batch_count+=1

    return total_loss/batch_count


@torch.no_grad()
def evaluate(model, testLoader):
    b=0
    acc=0

    for batch in testLoader: 
        x, y = batch
        x = x.to('cuda')
        out  = model(x)
        out = torch.argmax(input=out, dim=-1)
        acc+=accuracy_score(y, out.cpu())
        b+=1
    
    return acc/b


def prepare_data():
    """
    Not implemented as DDP
    """
    print("Preparing data")

    training_data = datasets.CIFAR10(
        root="data",
        train=True,
        download=True,
        transform=ToTensor()
    )

    test_data = datasets.CIFAR10(
        root="data",
        train=False,
        download=True,
        transform=ToTensor()
    )

    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

    return train_dataloader, test_dataloader


def prepare_model(rank):
    print("Preparing model for process on GPU: ",rank)
    outputfn = nn.Softmax(dim=1)
    criterion = nn.CrossEntropyLoss()
    
    model = resnet('cifar10', 20)
    # model.fc = nn.Linear(512,10)  #10 classes in CIFAR10
    model.add_module(name="ouputFN", module=outputfn)
    
    return model, criterion


def run():
    nprocs = torch.cuda.device_count()
    trainLoader, testLoader =  prepare_data()
    barrier = mp.Barrier(nprocs)
    
    #Wandb Params
    project_name = ""
    entity_name = ""
    name = ""
    optim_choice = "DINSGD"

    #Algorithm Hyperparams
    if optim_choice == "DINSGD":
        p=0.001
        q=0.5
        beta=0.9
        optim_params = {"optim":"DINSGD", "p":p, "q":q, "beta":beta, "rho": 2}

    elif optim_choice == "SGD":
        lr = 0.1
        weight_decay = 0 #5*1e-4
        momentum = 0 #0.9

        optim_params = {"optim":"SGD" ,"lr":lr, "weight_decay":weight_decay, "momentum":momentum}
    elif optim_choice == "ADAM":
        lr = 0.0001
        weight_decay = 1e-6
        eps = 1e-3

        optim_params = {"optim":"ADAM" ,"lr":lr,"weight_decay":weight_decay,"eps":eps}
    else:
        raise("Optimizer not defined")
    
  
    args=[{
        "world_size":nprocs,
        "backend": "nccl",
        "init_method": "tcp://127.0.0.1:21357", #21457
        "name": "Compressed",
        "epochs":100,
        "trainLoader":trainLoader,
        "testLoader":testLoader,
        "barrier": barrier,
        "optim_params": optim_params,
        "compression_method":"quantize",
        "project_name":project_name,
        "entity_name":entity_name,
        "name":name

    }]
    
    mp.spawn(fn=allocate_process_to_GPU, args=args, join=True, nprocs=nprocs)


if __name__=='__main__':
    run()