import os
import argparse
import numpy as np
from tensorboard.backend.event_processing import event_accumulator
from matplotlib import pyplot as plt

# 显示中文的功能，如果不需要，可以移除这部分
from matplotlib.font_manager import FontProperties


def getChineseFont():  # 设置中文字体
    """设置中文字体的路径，Windows 的常用字体路径如下"""
    return FontProperties(fname='C:/Windows/Fonts/simhei.ttf')


def smooth(target, weight):  # 平滑处理函数
    """使用指数加权平均对数据进行平滑处理"""
    smoothed = []
    last = target[0].value  # 初始值
    for i in target:
        smoothed_val = {'step': i.step, 'value': last * weight + (1 - weight) * i.value}
        smoothed.append(smoothed_val)
        last = smoothed_val['value']
    return smoothed


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='')
    parser.add_argument('--folders', type=str, nargs='*', default=[

        'DARC',
        'PAR',
        'SAC',
        'SAC_IW',
        'VGDF',

    ])
    parser.add_argument('--color', type=str, nargs='*', default=['#FF0000', '#A52A2A', '#0077bb', '#FFA500', '#800080', ]) #'#cc3311', '#0077bb'
    parser.add_argument('--labels', type=str, nargs='*', default=['DARC', 'PAR', 'SAC', 'SAC_IW', 'VGDF'])
    parser.add_argument('--title', type=str, default='kinematic')
    parser.add_argument('--save_name', type=str, default='SR')
    parser.add_argument('--smooth', type=float, default=0.5)
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    plt.rcParams['savefig.dpi'] = 500  # 图片像素
    plt.rcParams['figure.dpi'] = 500  # 分辨率
    plt.figure()
    # plt.xlabel("Step", fontsize=14)
    plt.xlabel("步", fontsize=12,fontproperties=getChineseFont())
    # plt.ylabel("Average Reward", fontsize=14)
    plt.ylabel("标准化得分", fontsize=12,fontproperties=getChineseFont())

    plt.title(args.title, fontsize=14)
    lines = []
    for (folder, color) in zip(args.folders, args.color):
        log_path = os.path.join(args.path, folder)
        ea = event_accumulator.EventAccumulator(log_path)
        ea.Reload()
        test_rew = ea.scalars.Items('test/target_normalized_score')
        smoothed_rew = smooth(test_rew[:], args.smooth)
        rew_line, = plt.plot(
            list(map(lambda x: x['step'], smoothed_rew)),
            list(map(lambda x: x['value'], smoothed_rew)),
            color=color
        )
        lines.append(rew_line)

    plt.legend(handles=lines,
               labels=args.labels,
               loc='lower right')
    plt.grid(ls='--')  # 网格
    plt.savefig('ns.png')
