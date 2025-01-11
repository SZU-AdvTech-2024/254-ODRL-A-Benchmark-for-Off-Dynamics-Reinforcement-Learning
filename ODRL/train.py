import numpy as np  # 数值计算
import torch  # 深度学习
import gym  # 强化学习环境
import argparse  # 命令行解析
import os  # 路径管理，文件夹管理
import random  # 随机种子
import math  # 数学运算
import time  # 时间戳
import copy  # 深拷贝
import yaml  # 解析 YAML 格式，加载超参数设置
import json  # in case the user want to modify the hyperparameters 动态调整超参数
import d4rl # used to make offline environments for source domains 强化学习离线数据集和环境
import algo.utils as utils  # 自定义：强化学习工具函数

from algo.call_algo import call_algo  # 自定义：调用不同的强化学习算法
from dataset.call_dataset import call_tar_dataset  # 自定义：调用不同的强化学习数据集
from envs.mujoco.call_mujoco_env import call_mujoco_env  # 自定义：构建 MuJoCo 环境
from envs.adroit.call_adroit_env import call_adroit_env  # 自定义：构建 Adroit 环境
from envs.antmaze.call_antmaze_env import call_antmaze_env  # 自定义：构建 AntMaze 环境
from envs.infos import get_normalized_score  # 获取标准化后的奖励
from pathlib import Path  # 跨平台操作文件路径，管理日志、模型
from tensorboardX import SummaryWriter  # 记录训练过程中的指标，TensorBoard 中可视化分析


def eval_policy(policy, env, eval_episodes=10, eval_cnt=None):  # 评估策略的平均奖励
    avg_reward = 0.  # 初始化平均奖励
    for episode_idx in range(eval_episodes):  # 评估的回合数
        state, done = env.reset(), False  # 回合开始时重置环境
        while not done:  # 每回合流程
            action = policy.select_action(np.array(state))
            next_state, reward, done, _ = env.step(action)

            avg_reward += reward
            state = next_state
    avg_reward /= eval_episodes  # 计算评价奖励

    print("[{}] Evaluation over {} episodes: {}".format(eval_cnt, eval_episodes, avg_reward))
    # 打印：【当前回合】 评估{总回合数}下：{平均奖励}
    return avg_reward


if __name__ == "__main__":
    parser = argparse.ArgumentParser()  # 命令行参数解析器
    parser.add_argument("--dir", default="./logs")  # 日志文件存储路径，默认/logs
    parser.add_argument("--policy", default="SAC", help='policy to use')  # 强化学习算法
    parser.add_argument("--env", default="halfcheetah-friction")  # 强化学习环境
    parser.add_argument('--srctype', default="medium", help='dataset type used in the source domain')
    # only useful when source domain is offline 指定源领域数据集类型
    parser.add_argument('--tartype', default="medium", help='dataset type used in the target domain')
    # only useful when target domain is offline 指定目标领域数据集类型
    # support dataset type:支持数据集类型：
    # source domain: all valid datasets from D4RL          源域：D4RL 提供的所有合法数据集
    # target domain: random, medium, medium-expert, expert 目标域： random、medium、medium-expert、expert 等
    parser.add_argument('--shift_level', default=0.1, help='the scale of the dynamics shift. Note that this value varies on different settins')
    # 环境动态偏移程度
    parser.add_argument('--mode', default=0, type=int, help='the training mode, there are four types, 0: online-online, 1: offline-online, 2: online-offline, 3: offline-offline')
    # 训练模式：0：在线-在线，1：离线-在线，2：在线-离线，3：离线-离线
    parser.add_argument("--seed", default=0, type=int)  # 随机种子
    parser.add_argument("--save-model", action="store_true")        # Save model and optimizer parameters
    # 是否保存模型参数，默认情况下此参数为 False，加上该选项后为 True
    parser.add_argument('--tar_env_interact_interval', help='interval of interacting with target env', default=10, type=int)
    # 与目标环境交互的时间间隔（以训练步数为单位）
    parser.add_argument('--max_step', default=int(1e6), type=int)  # the maximum gradient step for off-dynamics rl learning
    # 最大训练步数
    parser.add_argument('--params', default=None, help='Hyperparameters for the adopted algorithm, ought to be in JSON format')
    # 指定算法的超参数，需以 JSON 格式传入
    args = parser.parse_args()  # 解析命令行参数

    # we support different ways of specifying tasks, e.g., hopper-friction, hopper_friction, hopper_morph_torso_easy, hopper-morph-torso-easy
    # 支持下划线和横杠混用任务命名格式
    if '_' in args.env:
        args.env = args.env.replace('_', '-')

    # ant, halfcheetah, hopper, walker2d这四种机器人的Locomotion任务，任务域：mujoco
    if 'halfcheetah' in args.env or 'hopper' in args.env or 'walker2d' in args.env or args.env.split('-')[0] == 'ant':
        domain = 'mujoco'
    # pen, door, relocate, hammer这四种Dexterous Manipulation任务，任务域：adroit
    elif 'pen' in args.env or 'relocate' in args.env or 'door' in args.env or 'hammer' in args.env:
        domain = 'adroit'
    elif 'antmaze' in args.env:
        domain = 'antmaze'
    else:
        raise NotImplementedError
    print(domain)

    call_env = {
        'mujoco': call_mujoco_env,
        'adroit': call_adroit_env,
        'antmaze': call_antmaze_env,
    }

    # determine referenced environment name
    ref_env_name = args.env + '-' + str(args.shift_level)
    
    if domain == 'antmaze':
        src_env_name = args.env
        src_env_name_config = args.env
    elif domain == 'adroit':
        src_env_name = args.env
        src_env_name_config = args.env.split('-')[0]
    else:
        src_env_name = args.env.split('-')[0]
        src_env_name_config = src_env_name
    tar_env_name = args.env

    # make environments
    if args.mode == 1 or args.mode == 3:
        if domain == 'antmaze':
            src_env_name = src_env_name.split('-')[0]
            src_env_name += '-' + args.srctype + '-v0'
        elif domain == 'adroit':
            src_env_name = src_env_name.split('-')[0]
            src_env_name += '-' + args.srctype + '-v0'
        else:
            src_env_name += '-' + args.srctype + '-v2'
        src_env = None
        src_eval_env = gym.make(src_env_name)
        src_eval_env.seed(args.seed)
    else:
        if 'antmaze' in src_env_name:
            src_env_config = {
                'env_name': src_env_name,
                'shift_level': args.shift_level,
            }
            src_env = call_env[domain](src_env_config)
            src_env.seed(args.seed)
            src_eval_env = call_env[domain](src_env_config)
            src_eval_env.seed(args.seed + 100)
        else:
            src_env_config = {
                'env_name': src_env_name,
                'shift_level': args.shift_level,
            }
            src_env = call_env[domain](src_env_config)
            src_env.seed(args.seed)
            src_eval_env = copy.deepcopy(src_env)
            src_eval_env.seed(args.seed + 100)

    if args.mode == 2 or args.mode == 3:
        tar_env = None
        tar_env_config = {
            'env_name': tar_env_name,
            'shift_level': args.shift_level,
        }
        tar_eval_env = call_env[domain](tar_env_config)
        tar_eval_env.seed(args.seed + 100)
    else:
        tar_env_config = {
            'env_name': tar_env_name,
            'shift_level': args.shift_level,
        }
        tar_env = call_env[domain](tar_env_config)
        tar_env.seed(args.seed)
        tar_eval_env = call_env[domain](tar_env_config)
        tar_eval_env.seed(args.seed + 100)
    
    if args.mode not in [0,1,2,3]:
        raise NotImplementedError # cannot support other modes
    
    policy_config_name = args.policy.lower()

    # load pre-defined hyperparameter config for training
    with open(f"{str(Path(__file__).parent.absolute())}/config/{domain}/{policy_config_name}/{src_env_name_config}.yaml", 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    if args.params is not None:
        override_params = json.loads(args.params)
        config.update(override_params)
        print('The following parameters are updated to:', args.params)

    print("------------------------------------------------------------")
    print("Policy: {}, Env: {}, Seed: {}".format(args.policy, args.env, args.seed))
    print("------------------------------------------------------------")
    
    # log path, we use logging with tensorboard
    if args.mode == 1:
        outdir = args.dir + '/' + args.policy + '/' + args.env + '-srcdatatype-' + args.srctype + '-' + str(args.shift_level) + '/r' + str(args.seed)
    elif args.mode == 2:
        outdir = args.dir + '/' + args.policy + '/' + args.env + '-tardatatype-' + args.tartype + '-' + str(args.shift_level) + '/r' + str(args.seed)
    elif args.mode == 3:
        outdir = args.dir + '/' + args.policy + '/' + args.env + '-srcdatatype-' + args.srctype + '-tardatatype-' + args.tartype + '-' + str(args.shift_level) + '/r' + str(args.seed)
    else:
        outdir = args.dir + '/' + args.policy + '/' + args.env + '-' + str(args.shift_level) + '/r' + str(args.seed)
    writer = SummaryWriter('{}/tb'.format(outdir))
    if args.save_model and not os.path.exists("{}/models".format(outdir)):
        os.makedirs("{}/models".format(outdir))

    # seed all
    src_env.action_space.seed(args.seed) if src_env is not None else None
    tar_env.action_space.seed(args.seed) if tar_env is not None else None
    src_eval_env.action_space.seed(args.seed)
    tar_eval_env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)

    # get necessary information from both domains
    state_dim = src_eval_env.observation_space.shape[0]
    action_dim = src_eval_env.action_space.shape[0] 
    max_action = float(src_eval_env.action_space.high[0])
    min_action = -max_action
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # determine shift_level
    if domain == 'mujoco':
        if args.shift_level in ['easy', 'medium', 'hard']:
            shift_level = args.shift_level
        else:
            shift_level = float(args.shift_level)
    else:
        shift_level = args.shift_level

    config.update({
        'env_name': args.env,
        'state_dim': state_dim,
        'action_dim': action_dim,
        'max_action': max_action,
        'tar_env_interact_interval': int(args.tar_env_interact_interval),
        'max_step': int(args.max_step),
        'shift_level': shift_level,
    })

    policy = call_algo(args.policy, config, args.mode, device)
    
    ## write logs to record training parameters
    with open(outdir + 'log.txt','w') as f:
        f.write('\n Policy: {}; Env: {}, seed: {}'.format(args.policy, args.env, args.seed))
        for item in config.items():
            f.write('\n {}'.format(item))

    src_replay_buffer = utils.ReplayBuffer(state_dim, action_dim, device)
    tar_replay_buffer = utils.ReplayBuffer(state_dim, action_dim, device)

    # in case that the domain is offline, we directly load its offline data
    if args.mode == 1 or args.mode == 3:
        src_replay_buffer.convert_D4RL(d4rl.qlearning_dataset(src_eval_env))
        if 'antmaze' in args.env:
            src_replay_buffer.reward -= 1.0
    
    if args.mode == 2 or args.mode == 3:
        tar_dataset = call_tar_dataset(tar_env_name, shift_level, args.tartype)
        tar_replay_buffer.convert_D4RL(tar_dataset)
        if 'antmaze' in args.env:
            tar_replay_buffer.reward -= 1.0

    eval_cnt = 0
    
    eval_src_return = eval_policy(policy, src_eval_env, eval_cnt=eval_cnt)
    eval_tar_return = eval_policy(policy, tar_eval_env, eval_cnt=eval_cnt)
    eval_cnt += 1

    if args.mode == 0:
        # online-online learning

        src_state, src_done = src_env.reset(), False
        tar_state, tar_done = tar_env.reset(), False
        src_episode_reward, src_episode_timesteps, src_episode_num = 0, 0, 0
        tar_episode_reward, tar_episode_timesteps, tar_episode_num = 0, 0, 0

        for t in range(int(config['max_step'])):
            src_episode_timesteps += 1

            # select action randomly or according to policy, if the policy is deterministic, add exploration noise akin to TD3 implementation
            src_action = (
                policy.select_action(np.array(src_state), test=False) + np.random.normal(0, max_action * 0.2, size=action_dim)
            ).clip(-max_action, max_action)

            src_next_state, src_reward, src_done, _ = src_env.step(src_action) 
            src_done_bool = float(src_done) if src_episode_timesteps < src_env._max_episode_steps else 0

            if 'antmaze' in args.env:
                src_reward -= 1.0

            src_replay_buffer.add(src_state, src_action, src_next_state, src_reward, src_done_bool)

            src_state = src_next_state
            src_episode_reward += src_reward
            
            # interaction with tar env
            if t % config['tar_env_interact_interval'] == 0:
                tar_episode_timesteps += 1
                tar_action = policy.select_action(np.array(tar_state), test=False)

                tar_next_state, tar_reward, tar_done, _ = tar_env.step(tar_action)
                tar_done_bool = float(tar_done) if tar_episode_timesteps < src_env._max_episode_steps else 0

                if 'antmaze' in args.env:
                    tar_reward -= 1.0

                tar_replay_buffer.add(tar_state, tar_action, tar_next_state, tar_reward, tar_done_bool)

                tar_state = tar_next_state
                tar_episode_reward += tar_reward

            policy.train(src_replay_buffer, tar_replay_buffer, config['batch_size'], writer)
            
            if src_done: 
                print("Total T: {} Episode Num: {} Episode T: {} Reward: {}".format(t+1, src_episode_num+1, src_episode_timesteps, src_episode_reward))
                writer.add_scalar('train/source return', src_episode_reward, global_step = t+1)

                src_state, src_done = src_env.reset(), False
                src_episode_reward = 0
                src_episode_timesteps = 0
                src_episode_num += 1
            
            if tar_done:
                print("Total T: {} Episode Num: {} Episode T: {} Reward: {}".format(t+1, tar_episode_num+1, tar_episode_timesteps, tar_episode_reward))
                writer.add_scalar('train/target return', tar_episode_reward, global_step = t+1)
                # record normalized score
                train_normalized_score = get_normalized_score(tar_episode_reward, ref_env_name)
                writer.add_scalar('train/target normalized score', train_normalized_score, global_step = t+1)

                tar_state, tar_done = tar_env.reset(), False
                tar_episode_reward = 0
                tar_episode_timesteps = 0
                tar_episode_num += 1

            if (t + 1) % config['eval_freq'] == 0:
                src_eval_return = eval_policy(policy, src_eval_env, eval_cnt=eval_cnt)
                tar_eval_return = eval_policy(policy, tar_eval_env, eval_cnt=eval_cnt)
                writer.add_scalar('test/source return', src_eval_return, global_step = t+1)
                writer.add_scalar('test/target return', tar_eval_return, global_step = t+1)
                # record normalized score
                eval_normalized_score = get_normalized_score(tar_eval_return, ref_env_name)
                writer.add_scalar('test/target normalized score', eval_normalized_score, global_step = t+1)

                eval_cnt += 1

                if args.save_model:
                    policy.save('{}/models/model'.format(outdir))
    elif args.mode == 1:
        # offline-online learning
        tar_state, tar_done = tar_env.reset(), False
        tar_episode_reward, tar_episode_timesteps, tar_episode_num = 0, 0, 0

        for t in range(int(config['max_step'])):
            
            # interaction with tar env
            if t % config['tar_env_interact_interval'] == 0:
                tar_episode_timesteps += 1
                tar_action = policy.select_action(np.array(tar_state), test=False)

                tar_next_state, tar_reward, tar_done, _ = tar_env.step(tar_action)
                tar_done_bool = float(tar_done) if tar_episode_timesteps < src_eval_env._max_episode_steps else 0

                if 'antmaze' in args.env:
                    tar_reward -= 1.0

                tar_replay_buffer.add(tar_state, tar_action, tar_next_state, tar_reward, tar_done_bool)

                tar_state = tar_next_state
                tar_episode_reward += tar_reward

            policy.train(src_replay_buffer, tar_replay_buffer, config['batch_size'], writer)
            
            if tar_done:
                print("Total T: {} Episode Num: {} Episode T: {} Reward: {}".format(t+1, tar_episode_num+1, tar_episode_timesteps, tar_episode_reward))
                writer.add_scalar('train/target return', tar_episode_reward, global_step = t+1)
                train_normalized_score = get_normalized_score(tar_episode_reward, ref_env_name)
                writer.add_scalar('train/target normalized score', train_normalized_score, global_step = t+1)

                tar_state, tar_done = tar_env.reset(), False
                tar_episode_reward = 0
                tar_episode_timesteps = 0
                tar_episode_num += 1

            if (t + 1) % config['eval_freq'] == 0:
                src_eval_return = eval_policy(policy, src_eval_env, eval_cnt=eval_cnt)
                tar_eval_return = eval_policy(policy, tar_eval_env, eval_cnt=eval_cnt)
                writer.add_scalar('test/source return', src_eval_return, global_step = t+1)
                writer.add_scalar('test/target return', tar_eval_return, global_step = t+1)
                eval_normalized_score = get_normalized_score(tar_eval_return, ref_env_name)
                writer.add_scalar('test/target normalized score', eval_normalized_score, global_step = t+1)

                eval_cnt += 1

                if args.save_model:
                    policy.save('{}/models/model'.format(outdir))
    elif args.mode == 2:
        # online-offline learning
        src_state, src_done = src_env.reset(), False
        src_episode_reward, src_episode_timesteps, src_episode_num = 0, 0, 0

        for t in range(int(config['max_step'])):
            src_episode_timesteps += 1

            # select action randomly or according to policy, if the policy is deterministic, add exploration noise akin to TD3 implementation
            src_action = (
                policy.select_action(np.array(src_state), test=False) + np.random.normal(0, max_action * 0.2, size=action_dim)
            ).clip(-max_action, max_action)

            src_next_state, src_reward, src_done, _ = src_env.step(src_action) 
            src_done_bool = float(src_done) if src_episode_timesteps < src_env._max_episode_steps else 0

            if 'antmaze' in args.env:
                src_reward -= 1.0

            src_replay_buffer.add(src_state, src_action, src_next_state, src_reward, src_done_bool)

            src_state = src_next_state
            src_episode_reward += src_reward

            policy.train(src_replay_buffer, tar_replay_buffer, config['batch_size'], writer)
            
            if src_done: 
                print("Total T: {} Episode Num: {} Episode T: {} Reward: {}".format(t+1, src_episode_num+1, src_episode_timesteps, src_episode_reward))
                writer.add_scalar('train/source return', src_episode_reward, global_step = t+1)

                src_state, src_done = src_env.reset(), False
                src_episode_reward = 0
                src_episode_timesteps = 0
                src_episode_num += 1

            if (t + 1) % config['eval_freq'] == 0:
                src_eval_return = eval_policy(policy, src_eval_env, eval_cnt=eval_cnt)
                tar_eval_return = eval_policy(policy, tar_eval_env, eval_cnt=eval_cnt)
                writer.add_scalar('test/source return', src_eval_return, global_step = t+1)
                writer.add_scalar('test/target return', tar_eval_return, global_step = t+1)
                eval_normalized_score = get_normalized_score(tar_eval_return, ref_env_name)
                writer.add_scalar('test/target normalized score', eval_normalized_score, global_step = t+1)

                eval_cnt += 1

                if args.save_model:
                    policy.save('{}/models/model'.format(outdir))
    else:
        # offline-offline learning
        for t in range(int(config['max_step'])):
            policy.train(src_replay_buffer, tar_replay_buffer, config['batch_size'], writer)

            if (t + 1) % config['eval_freq'] == 0:
                src_eval_return = eval_policy(policy, src_eval_env, eval_cnt=eval_cnt)
                tar_eval_return = eval_policy(policy, tar_eval_env, eval_cnt=eval_cnt)
                writer.add_scalar('test/source return', src_eval_return, global_step = t+1)
                writer.add_scalar('test/target return', tar_eval_return, global_step = t+1)
                eval_normalized_score = get_normalized_score(tar_eval_return, ref_env_name)
                writer.add_scalar('test/target normalized score', eval_normalized_score, global_step = t+1)
                
                eval_cnt += 1

                if args.save_model:
                    policy.save('{}/models/model'.format(outdir))
    writer.close()
