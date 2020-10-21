import argparse
import time
import os

from sklearn.metrics import confusion_matrix

from dataset import I3DDataSet
from models import i3d
from transforms import *

# options
parser = argparse.ArgumentParser(description="Standard video-level testing")
parser.add_argument('dataset', type=str, choices=['ucf101', 'hmdb51', 'kinetics'])
parser.add_argument('modality', type=str, choices=['RGB', 'Flow'])
parser.add_argument('root_path', type=str)
parser.add_argument('test_list', type=str)
parser.add_argument('weights', type=str)
parser.add_argument('--arch', type=str, default='i3d_resnet50')
parser.add_argument('--save_scores', type=str, default=None)
parser.add_argument('--max_num', type=int, default=-1)
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('--clip_length', default=250, type=int, metavar='N',
                    help='length of sequential frames (default: 64)')
parser.add_argument('--dropout', type=float, default=0.7)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--gpus', nargs='+', type=int, default=None)
parser.add_argument('--flow_prefix', type=str, default='flow_')

args = parser.parse_args()

if args.dataset == 'ucf101':
    num_classes = 101
elif args.dataset == 'hmdb51':
    num_classes = 51
elif args.dataset == 'kinetics':
    num_classes = 400
else:
    raise ValueError('Unknown dataset ' + args.dataset)

model = getattr(i3d, args.arch)(modality=args.modality, num_classes=num_classes,
                                dropout_ratio=args.dropout)

checkpoint = torch.load(args.weights)
print("model epoch {} best prec@1: {}".format(checkpoint['epoch'], checkpoint['best_prec1']))

base_dict = {'.'.join(k.split('.')[1:]): v for k, v in list(checkpoint['state_dict'].items())}
model.load_state_dict(base_dict)

# Data loading code
crop_size = args.input_size
scale_size = args.input_size * 256 // 224
input_mean = [0.485, 0.456, 0.406]
input_std = [0.229, 0.224, 0.225]
if args.modality == 'Flow':
    input_mean = [0.5]
    input_std = [np.mean(input_std)]

data_loader = torch.utils.data.DataLoader(
    I3DDataSet(args.root_path, args.test_list, clip_length=args.clip_length, modality=args.modality,
               image_tmpl="img_{:05d}.jpg" if args.modality == "RGB" else args.flow_prefix + "{}_{:05d}.jpg",
               transform=torchvision.transforms.Compose([
                   GroupScale(scale_size),
                   GroupCenterCrop(crop_size),
                   ToNumpyNDArray(),
                   ToTorchFormatTensor(),
                   GroupNormalize(input_mean, input_std),
               ]),
               test_mode=True),
    batch_size=1, shuffle=False,
    num_workers=args.workers * 2, pin_memory=True)

if args.gpus is not None:
    devices = [args.gpus[i] for i in range(args.workers)]
else:
    devices = list(range(args.workers))

model = torch.nn.DataParallel(model.cuda())
model.eval()

data_gen = enumerate(data_loader)

total_num = len(data_loader.dataset)
output = []


def eval_video(video_data):
    i, data, label = video_data

    rst = model(data).data.cpu().numpy().copy()
    return i, rst, label[0]


proc_start_time = time.time()
max_num = args.max_num if args.max_num > 0 else len(data_loader.dataset)

for i, (data, label) in data_gen:
    if i >= max_num:
        break
    rst = eval_video((i, data, label))
    output.append(rst[1:])
    cnt_time = time.time() - proc_start_time
    if i % 10 == 0:
        print('video {} done, total {}/{}, average {} sec/video'.format(i, i + 1,
                                                                        total_num,
                                                                        float(cnt_time) / (i + 1)))

video_pred = [np.argmax(x[0]) for x in output]

video_labels = [x[1] for x in output]

cf = confusion_matrix(video_labels, video_pred).astype(float)

cls_cnt = cf.sum(axis=1)
cls_hit = np.diag(cf)

cls_acc = cls_hit / cls_cnt

print(cls_acc)

print('Accuracy {:.02f}%'.format(np.mean(cls_acc) * 100))

if args.save_scores is not None:

    # reorder before saving
    name_list = [x.strip().split()[0] for x in open(args.test_list)]

    order_dict = {e: i for i, e in enumerate(sorted(name_list))}

    reorder_output = [None] * len(output)
    reorder_label = [None] * len(output)

    for i in range(len(output)):
        idx = order_dict[name_list[i]]
        reorder_output[idx] = output[i]
        reorder_label[idx] = video_labels[i]

    np.savez(args.save_scores, scores=reorder_output, labels=reorder_label)
