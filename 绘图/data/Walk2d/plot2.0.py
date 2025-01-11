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

        'H2O',
        'PAR_BC',
        'SAC_BC',
        'SAC_CQL',

    ])
    parser.add_argument('--color', type=str, nargs='*',
                        default=['#ff7f0e', '#008080', '#3357FF', '#FF33A1'])  # '#cc3311', '#0077bb'
    parser.add_argument('--labels', type=str, nargs='*', default=['H2O', 'PAR_BC', 'SAC_BC', 'SAC_CQL'])
    parser.add_argument('--title', type=str, default='ANT_F5')
    parser.add_argument('--save_name', type=str, default='SR')
    parser.add_argument('--smooth', type=float, default=0.5)
    return parser.parse_args()


def calculate_std(target):
    """计算每个步骤的标准差"""
    std_list = []
    last_step = target[0].step
    values = []
    for i in target:
        if i.step != last_step:
            std_list.append({'step': last_step, 'std': np.std(values)})
            last_step = i.step
            values = [i.value]
        else:
            values.append(i.value)
    if values:  # 最后一组数据
        std_list.append({'step': last_step, 'std': np.std(values)})
    return std_list


if __name__ == '__main__':
    args = get_args()
    plt.rcParams['savefig.dpi'] = 500  # 图片像素
    plt.rcParams['figure.dpi'] = 500  # 分辨率
    plt.figure()
    plt.xlabel("步", fontsize=12, fontproperties=getChineseFont())
    plt.ylabel("奖励", fontsize=12, fontproperties=getChineseFont())
    plt.title(args.title, fontsize=14)
    lines = []

    for (folder, color) in zip(args.folders, args.color):
        log_path = os.path.join(args.path, folder)
        ea = event_accumulator.EventAccumulator(log_path)
        ea.Reload()
        test_rew = ea.scalars.Items('test/source_return')

        smoothed_rew = smooth(test_rew[:], args.smooth)

        # 计算标准差
        std_rew = calculate_std(test_rew[:])

        # 打印平滑后的奖励和标准差以供调试
        print(f"Folder: {folder}")
        print("Smoothed Rewards (step, value):")
        print(list(map(lambda x: (x['step'], x['value']), smoothed_rew)))
        print("Standard Deviations (step, std):")
        print(list(map(lambda x: (x['step'], x['std']), std_rew)))

        # 绘制平滑后的奖励曲线
        rew_line, = plt.plot(
            list(map(lambda x: x['step'], smoothed_rew)),
            list(map(lambda x: x['value'], smoothed_rew)),
            color=color
        )

        # 计算标准差区间
        steps = list(map(lambda x: x['step'], smoothed_rew))
        smoothed_values = np.array(list(map(lambda x: x['value'], smoothed_rew)))
        std_values = np.array(list(map(lambda x: x['std'], std_rew)))

        # 确保步数和标准差数据的对齐
        if len(steps) != len(smoothed_values) or len(steps) != len(std_values):
            print(f"Warning: The lengths of smoothed values and standard deviations do not match for folder {folder}")

        # 绘制标准差阴影区域
        plt.fill_between(
            steps,
            smoothed_values - std_values,
            smoothed_values + std_values,
            color=color, alpha=0.3  # 设置透明度
        )

        lines.append(rew_line)

    plt.legend(handles=lines,
               labels=args.labels,
               loc='lower right')
    plt.grid(ls='--')  # 网格
    plt.savefig('SR.png')

