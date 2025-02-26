'''
@author: Chen Dong
HIT

'''
import random
import scipy.io as sio
from torch.utils.data import DataLoader
from utils import plots
import argparse
from utils.utils import *
from utils.data_read import data_get_func
from models.UNmamba import UNMambaLinear


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def seed_torch(seed=1):
    '''
    Keep the seed fixed thus the results can keep stable
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# ---------------defined paramaters------------------ #
# # Setting Params
parser = argparse.ArgumentParser(description='Training for HSI Unmixing')
parser.add_argument('-d', '--dataset', dest='dataset', choices=['S1', 'JR', 'UR', 'UR5', 'UR6', 'SA', 'AP'],
                    default='AP', help="Name of dataset.")
parser.add_argument('-i', '--iter', type=int, dest='iter', default=1, help="No of Monte Carlo test")
parser.add_argument('-e', '--epochs', type=int, default=1000, help='epoch number')
parser.add_argument('-ds', '--down_ratio', type=int, default=16, help='down ratio')
parser.add_argument('-nq', '--num_queries', type=int, default=50, help='query number')
parser.add_argument('-b', '--batch_size', type=int, default=1, help='number of batch size')
parser.add_argument('-n', '--num_copies', type=int, default=1, help='number of copies')
parser.add_argument('--model_name', type=str, default='UNMamba', help='model used')
parser.add_argument('--seed', type=int, default=11, help='number of seed')

parser.add_argument('-l', '--lr', type=float, default=3e-3, help='learning rate')
parser.add_argument('--weight_mse', type=float, default=10, help='weight_mse')
parser.add_argument('--weight_sad', type=float, default=1, help='weight_sad')
parser.add_argument('--weight_sid', type=float, default=0, help='weight_sid')
parser.add_argument('--weight_endm', type=float, default=1, help='weight_endm')
parser.add_argument('--weight_aban', type=float, default=1e-2, help='weight_endm')

parser.add_argument('--dropout', type=float, default=1e-2, help='weight_endm')

parser.add_argument('--weight_decay', type=float, default=0, help='weight_decay')
args = parser.parse_args()


seed_torch(seed=args.seed)
# define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# create saving path
dataset_query = {"AP": "Apex", "JR": "Japser Ridge", "SA": "Samson", "UR": "Urban", "UR5": "Urban5", "UR6": "Urban6"}
workspace = os.path.abspath(".")
file_path = os.path.join(workspace, 'dataset', dataset_query[args.dataset])
save_path = os.path.join('results', args.model_name, dataset_query[args.dataset])

if not os.path.exists(save_path):
    os.makedirs(save_path)

model_save_path = os.path.join(save_path, 'model')
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)

fig_save_path = os.path.join(save_path, 'index_fig')
if not os.path.exists(fig_save_path):
    os.makedirs(fig_save_path)

# 将预设参数保存为txt文件
with open(os.path.join(save_path, 'argparser_params.txt'), 'w') as f:
    for arg in vars(args):
        f.write(f"{arg}: {getattr(args, arg)}\n")

data_get = data_get_func(dataname=dataset_query[args.dataset], file_path=file_path)
data_hsi_img = data_get.data_hsi.reshape(data_get.num_bands, data_get.num_rows, data_get.num_cols).transpose((0, 2, 1))
data_aban_img = data_get.data_aban.reshape(data_get.num_endm, data_get.num_rows, data_get.num_cols).transpose((0, 2, 1))
print(data_hsi_img.shape)

# shape: 1x198x100x100
torch_hsi = torch.from_numpy(data_hsi_img).to(torch.float32).to(device).unsqueeze(0).repeat(args.num_copies, 1, 1, 1)
torch_aban = torch.from_numpy(data_aban_img).to(torch.float32).to(device)

train_data_loader = DataLoader(torch_hsi, batch_size=args.batch_size)

model = UNMambaLinear(num_band=data_get.num_bands, d_model=64,
                      num_endm=data_get.num_endm, num_queries_times=args.num_queries, ds=args.down_ratio,
                      dropout=args.dropout
                      ).to(device)

my_loss = My_Loss(weight_mse=args.weight_mse,
                  weight_sad=args.weight_sad,
                  weight_endm=args.weight_endm)

data_get = data_get_func(dataname=dataset_query[args.dataset], file_path=file_path)
data_hsi_img = data_get.data_hsi.reshape(data_get.num_bands, data_get.num_rows, data_get.num_cols).transpose((0, 2, 1))
data_aban_img = data_get.data_aban.reshape(data_get.num_endm, data_get.num_rows, data_get.num_cols).transpose((0, 2, 1))
torch_hsi = torch.from_numpy(data_hsi_img).to(torch.float32).to(device).unsqueeze(0).repeat(args.num_copies, 1, 1, 1)
torch_aban = torch.from_numpy(data_aban_img).to(torch.float32).to(device)
hsi_mean_tensor = torch.from_numpy(data_get.get_hsi_mean()).to(torch.float32).to(device)

train_data_loader = DataLoader(torch_hsi, batch_size=args.batch_size, shuffle=False)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=45, gamma=0.9)

model.train()

for epoch in range(0, args.epochs):
    loss_print = 0
    temp_i = 0
    for i, x in enumerate(train_data_loader):
        temp_i += 1
        pred_linear, pred_abun, pred_endm = model(x)
        loss = my_loss(x, pred_linear, pred_endm, hsi_mean_tensor, pred_aban=pred_abun)
        optimizer.zero_grad()
        loss.backward()
        for p in model.query_embed.weight:
            p.data.clamp_(1e-7, 1)
        optimizer.step()
        loss_print += loss
    loss_print = loss_print / (temp_i + 1)

    print("Epoch: %d/%d" % (epoch + 1, args.epochs),
          "| lr: %5f" % (optimizer.param_groups[0]['lr']),
          "| loss: %.4f" % loss_print.cpu().data.numpy()
          )

    scheduler.step()

model.eval()
torch.save(model.state_dict(), os.path.join(model_save_path, 'model.pth'))
test_hsi_img = data_hsi_img.copy()
test_aban_img = data_aban_img.copy().transpose((2, 1, 0))
test_endm = data_get.data_endm.copy()

torch_test_hsi = torch.from_numpy(test_hsi_img).to(torch.float32).to(device).unsqueeze(0)
re_linear, abu_est, endm_ = model(torch_test_hsi)

abu_est = abu_est.squeeze(0).permute(2, 1, 0).detach().cpu().numpy()
re_result = re_linear.squeeze(0).permute(2, 1, 0).detach().cpu().numpy()
est_endmem = model.get_endmember().detach().cpu().numpy()
est_endmem = est_endmem.T

index = [2, 0, 3, 1]
abu_est[:, :, np.arange(data_get.num_endm)] = abu_est[:, :, index]
est_endmem[:, np.arange(data_get.num_endm)] = est_endmem[:, index]

plots.plot_abundance(test_aban_img.transpose((1, 0, 2)), abu_est.transpose((1, 0, 2)),
                     data_get.num_endm,
                     save_dir=fig_save_path)
plots.plot_endmembers(data_get.data_endm, est_endmem, data_get.num_endm,
                      save_dir=fig_save_path)
