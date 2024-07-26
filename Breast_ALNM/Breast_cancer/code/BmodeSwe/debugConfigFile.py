import torch
from torchvision import transforms
from src_utils.utils import get_path

def getDebugConfig_BmodeSwe(trainOrtest):
    Brease_Cancer_pytorch_path_upDir = get_path(3)  # Get the absolute path of the parent directory
    if (trainOrtest == "test"):
        weights = ""
        save_weights_path= ""

        fusion_layer = 6

        trained_weight_path = Brease_Cancer_pytorch_path_upDir + "/Weights/test.pth"
        data_version = "Version_sample"



    else :

        weights = Brease_Cancer_pytorch_path_upDir + '/Weights/model_newdict_changed.pth'
        save_weights_path = Brease_Cancer_pytorch_path_upDir + '/Save_weights'
        trained_weight_path = " "

        fusion_layer = 6

        data_version = "Version_sample"

    model = "MBT_in21k_bmodeswe"
    data_transform = {
        "train": transforms.Compose([transforms.Resize(224),
                                     transforms.RandomHorizontalFlip(),  # 随机翻转，数据增强
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    return model,\
           device,\
           data_transform,\
           data_version,\
           fusion_layer,\
           save_weights_path,\
           weights,\
           trained_weight_path


