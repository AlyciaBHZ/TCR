# train.py for psiCLM with test eval and attention visualization

import random
import numpy as np
import torch
import os, sys, re
import argparse
import math
import torch.optim as opt
from torch.nn import functional as F

import data_clp as data
from model import psiCLM
from monte_carlo import psiMonteCarloSampler
from attn_visualize import visualize_attention
from model import analyze_mask_distribution

Batch_size = 640          # 大幅减小批次
ACCUMULATION_STEP = 4    # 减小累积步数，有效批次 = 64×4 = 256
TEST_STEP = 25          # 降低测试频率
VISION_STEP = 10

def get_device():
    # print("cuda:", torch.cuda.is_available())
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parse command-line arguments for conditioning
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--condition_set', type=int, choices=range(1, 9), default=1,
                    help='Condition set to use (1-8)')
parser.add_argument('--loss_type', type=str, choices=['standard', 'composite'], default='standard',
                    help='Loss function type: standard NLL or composite psi loss')
# 移除mask_strategy参数，简化配置
parser.add_argument('--use_monte_carlo', action='store_true',
                    help='Enable Monte Carlo collapse sampling')
parser.add_argument('--mc_frequency', type=int, default=10,
                    help='Apply Monte Carlo every N epochs')
parser.add_argument('--mc_weight', type=float, default=0.1,
                    help='Weight for Monte Carlo loss component')
# 🔧 新增分阶段训练参数
parser.add_argument('--staged_training', action='store_true',
                    help='Enable staged training: attention-only first, then full model')
parser.add_argument('--attention_only_epochs', type=int, default=30,
                    help='Number of epochs to train only attention parameters')

args = parser.parse_args()

print(f" Training Configuration:")
print(f"  - Condition set: {args.condition_set}")
print(f"  - Loss type: {args.loss_type}")
print(f"  - Masking: Input masking only (no attention constraints)")
print(f"  - Monte Carlo: {'yes' if args.use_monte_carlo else 'no'}")
if args.use_monte_carlo:
    print(f"    - MC frequency: every {args.mc_frequency} epochs")
    print(f"    - MC weight: {args.mc_weight}")
# 🔧 显示分阶段训练配置
if args.staged_training:
    print(f"  - Staged training: YES")
    print(f"    - Attention-only epochs: {args.attention_only_epochs}")
    print(f"    - Full model training starts from epoch {args.attention_only_epochs}")

condition_sets = {
    1: ['mhc', 'pep', 'lv', 'lj', 'hv', 'hj'],
    2: ['pep', 'lv', 'lj', 'hv', 'hj'],
    3: ['mhc', 'lv', 'lj', 'hv', 'hj'],
    4: ['lv', 'lj', 'hv', 'hj'],
    5: ['mhc', 'pep'],
    6: [],
    7: ['pep'],
}

conditioning_info = condition_sets[args.condition_set]
print('Using conditioning info:', conditioning_info)

train_set = data.CollapseProteinDataset('../data/trn.csv')
test_set  = data.CollapseProteinDataset('../data/tst.csv')

expdir = os.path.dirname(os.path.abspath(__file__))
if args.loss_type == 'composite':
    if args.staged_training:
        model_path = os.path.join(expdir, 'saved_model', f'staged_profile_condition_{args.condition_set}')
    else:
        model_path = os.path.join(expdir, 'saved_model', f'profile_condition_{args.condition_set}')
else:
    model_path = os.path.join(expdir, 'saved_model', f'condition_{args.condition_set}')
os.makedirs(model_path, exist_ok=True)

print(f"  - Model save path: {os.path.basename(model_path)}")

def test(model):
    model.eval()
    device = get_device()
    
    with torch.no_grad():
        losses = []
        for i in range(len(test_set)):
            sample = test_set[i]
            # 确保测试数据在正确设备上
            sample = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                     for k, v in sample.items()}
            
            #  关键修复：测试时创建全1的mask（不mask任何位置）
            sample_copy = sample.copy()
            hd_len = sample_copy['hd'].shape[0]
            sample_copy['mask'] = torch.ones(hd_len, device=device)  # 全1 = 不mask任何位置
            
            # 🔧 修复：根据训练时的loss类型来计算测试loss
            if args.loss_type == 'standard':
                loss = model(sample_copy, True, conditioning_info=conditioning_info)
            else:
                # 使用composite loss进行测试
                loss_dict = model.compute_composite_loss(sample_copy, conditioning_info)
                loss = loss_dict['total_loss']
            
            #  检查NaN
            if torch.isnan(loss) or torch.isinf(loss):
                print(f" Invalid loss at test sample {i}: {loss}")
                continue
                
            losses.append(loss.item())
            
    if not losses:
        return float('inf'), float('inf')
        
    avg_loss = sum(losses) / len(losses)
    pll = math.exp(avg_loss) if avg_loss < 10 else float('inf')  # 防止overflow
    return avg_loss, pll

def is_valid_sample(sample):
    try:
        # 检查HD序列不为空
        if sample['hd'].shape[0] == 0:
            return False
            
        # 检查mask不全为0
        if sample['mask'].sum() == 0:
            return False
            
        # 检查tensor类型
        if not sample['hd'].dtype.is_floating_point:
            return False
            
        return True
    except:
        return False

def freeze_non_attention_params(model):
    """冻结除attention外的所有参数"""
    frozen_count = 0
    total_count = 0
    for name, param in model.named_parameters():
        total_count += 1
        if 'attn' not in name and 'collapse' not in name:
            param.requires_grad = False
            frozen_count += 1
        else:
            param.requires_grad = True
    print(f"🔧 Frozen {frozen_count}/{total_count} parameters (keeping attention trainable)")

def unfreeze_all_params(model):
    """解冻所有参数"""
    for param in model.parameters():
        param.requires_grad = True
    print(f"🔧 All parameters unfrozen for full model training")

def train(model, optimizer, start_epoch, scheduler):
    mc_sampler = None
    if args.use_monte_carlo:
        class ModelLossFunction:
            def __init__(self, model):
                self.model = model
            
            def forward(self, pred_subset, target, mask, attn_traces, boundaries, sample):
                if args.loss_type == 'standard':
                    device = target.device
                    mask_expanded = mask[:, None] if mask.dim() == 1 else mask
                    pointwise_loss = -(pred_subset * target * mask_expanded[:len(pred_subset)]).sum(dim=-1)
                    total_loss = pointwise_loss.sum() / mask_expanded[:len(pred_subset)].sum()
                    return {'total_loss': total_loss}
                else:
                    return self.model.compute_composite_loss(sample, conditioning_info)
        
        model_loss_fn = ModelLossFunction(model)
        mc_sampler = psiMonteCarloSampler(model, model_loss_fn, cfg)

    best_pll = None
    better_count = 0
    current_epoch = start_epoch

    # 🔧 渐进式训练：动态调整正则化权重
    initial_collapse_weight = 0.01
    max_collapse_weight = 5.0  # 增加到5倍
    warmup_epochs = 100
    
    # 🔧 分阶段训练标志
    attention_training_phase = args.staged_training and current_epoch < args.attention_only_epochs
    if attention_training_phase:
        print(f"\n🎯 Starting ATTENTION-ONLY training phase (epochs 0-{args.attention_only_epochs})")
        freeze_non_attention_params(model)
    
    while True:
        model.train()
        
        # 🔧 检查是否需要切换训练阶段
        if args.staged_training and current_epoch == args.attention_only_epochs and attention_training_phase:
            print(f"\n🎯 Switching to FULL MODEL training phase (epoch {current_epoch}+)")
            unfreeze_all_params(model)
            attention_training_phase = False
            
            # 重新创建优化器以包含所有参数
            attention_params = []
            collapse_params = []
            other_params = []
            
            for name, param in model.named_parameters():
                if param.requires_grad:  # 只包含需要梯度的参数
                    if 'attn' in name:
                        attention_params.append(param)
                    elif 'collapse' in name:
                        collapse_params.append(param)
                    else:
                        other_params.append(param)
            
            optimizer = opt.Adam([
                {'params': other_params, 'lr': 5e-5, 'weight_decay': 1e-4},
                {'params': attention_params, 'lr': 1e-4, 'weight_decay': 1e-5},
                {'params': collapse_params, 'lr': 2e-4, 'weight_decay': 1e-6},
            ])
        
        # 🔧 动态调整正则化权重
        if args.loss_type == 'composite':
            if attention_training_phase:
                # 注意力训练阶段：使用更强的权重
                current_collapse_weight = 10.0  # 非常强的权重
            else:
                # 完整模型训练阶段：渐进式权重
                progress = min(1.0, (current_epoch - args.attention_only_epochs) / warmup_epochs) if args.staged_training else min(1.0, current_epoch / warmup_epochs)
                current_collapse_weight = initial_collapse_weight + (max_collapse_weight - initial_collapse_weight) * progress
            
            # 将权重传递给模型
            if hasattr(model, 'set_regularization_weights'):
                model.set_regularization_weights(current_collapse_weight)
            
            if current_epoch % 25 == 0:
                phase_name = "ATTENTION-ONLY" if attention_training_phase else "FULL MODEL"
                print(f"\n🔧 Epoch {current_epoch}: {phase_name} phase, Collapse weight = {current_collapse_weight:.1f}")
        
        batch_idxs = np.random.choice(len(train_set), Batch_size, replace=False)
        total_loss = 0
        loss_components = {'standard': 0, 'monte_carlo': 0, 'attention_only': 0}
        successful_samples = 0
        
        optimizer.zero_grad()

        for i, idx in enumerate(batch_idxs):
            sample = train_set[idx]
            device = get_device()
            sample = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                     for k, v in sample.items()}
            
            try:
                # 🔧 强制attention集中：每100个epoch进行一次attention重置
                if current_epoch % 100 == 0 and i == 0 and not attention_training_phase:
                    model._reset_attention_if_uniform()
                
                # 🔧 分阶段训练逻辑
                if attention_training_phase:
                    # 第一阶段：只训练attention，只使用熵损失
                    loss_dict = model.compute_composite_loss(sample, conditioning_info)
                    loss = -loss_dict['collapse_entropy']  # 直接最小化熵
                    loss_components['attention_only'] += loss.item()
                else:
                    # 第二阶段：完整模型训练
                    if args.loss_type == 'standard':
                        loss = model(sample, True, conditioning_info=conditioning_info)
                        loss_components['standard'] += loss.item()
                    else:
                        # 复合损失函数
                        loss_dict = model.compute_composite_loss(sample, conditioning_info)
                        loss = loss_dict['total_loss']
                        loss_components['standard'] += loss_dict['nll_loss'].item()
                
                # 数值检查
                if torch.isnan(loss) or torch.isinf(loss) or loss.item() < 1e-8:
                    continue
                
                # Monte Carlo增强（只在完整模型训练阶段）
                mc_loss = 0
                if mc_sampler and current_epoch % args.mc_frequency == 0 and not attention_training_phase:
                    try:
                        best_sample, best_energy, _ = mc_sampler.simulated_annealing(
                            sample, conditioning_info, n_steps=50)
                        
                        if args.loss_type == 'standard':
                            mc_loss = model(best_sample, True, conditioning_info=conditioning_info)
                        else:
                            mc_loss_dict = model.compute_composite_loss(best_sample, conditioning_info)
                            mc_loss = mc_loss_dict['total_loss']
                        
                        if not (torch.isnan(mc_loss) or torch.isinf(mc_loss)):
                            loss_components['monte_carlo'] += mc_loss.item()
                            loss = loss + args.mc_weight * mc_loss
                        
                    except Exception as e:
                        print(f" Monte Carlo step failed: {e}")
                
                loss.backward()
                total_loss += loss.item()
                successful_samples += 1
                
            except RuntimeError as e:
                if "does not require grad" in str(e):
                    continue
                else:
                    print(f" Unexpected error at sample {idx}: {e}")
                    continue

            if (i + 1) % ACCUMULATION_STEP == 0:
                optimizer.step()
                optimizer.zero_grad()

        avg_loss = total_loss / max(successful_samples, 1)

        # 损失组件记录
        if current_epoch % VISION_STEP == 0:
            sys.stdout.write('.')
            sys.stdout.flush()
            
            # 🔧 每个epoch随机采样一个用于调试
            if args.loss_type == 'composite' or attention_training_phase:
                try:
                    random_idx = random.randint(0, len(train_set) - 1)
                    debug_sample = train_set[random_idx]
                    debug_sample = {k: v.to(get_device()) if isinstance(v, torch.Tensor) else v 
                                   for k, v in debug_sample.items()}
                    with torch.no_grad():
                        debug_loss_dict = model.compute_composite_loss(debug_sample, conditioning_info)
                        if attention_training_phase:
                            print(f"E{current_epoch}: ATTN-ONLY Ent={debug_loss_dict['collapse_entropy'].item():.3f}, "
                                  f"Loss={-debug_loss_dict['collapse_entropy'].item():.3f}")
                        else:
                            print(f"E{current_epoch}: NLL={debug_loss_dict['nll_loss'].item():.3f}, "
                                  f"Ent={debug_loss_dict['collapse_entropy'].item():.3f}, "
                                  f"Tot={debug_loss_dict['total_loss'].item():.3f}")
                except:
                    pass

        if current_epoch % TEST_STEP == 0 and not attention_training_phase:  # 只在完整模型阶段测试
            tst_loss, tst_pll = test(model)
            print(f"Test E{current_epoch}: trn={avg_loss:.3f}, tst={tst_loss:.3f}, pll={tst_pll:.1f}")

            if not (math.isnan(tst_pll) or math.isinf(tst_pll)):
                if best_pll is None or tst_pll < best_pll:
                    best_pll = tst_pll
                    better_count = 0
                else:
                    better_count += TEST_STEP
            else:
                print(" Invalid test perplexity, continuing training...")

            if better_count >= 128:
                print("Early stopping: no improvement in perplexity.")
                break

            ckpt = os.path.join(model_path, f'model_epoch_{current_epoch}')
            torch.save(model.state_dict(), ckpt)
            torch.save(optimizer.state_dict(), ckpt + '.opt')

        current_epoch += 1

        # 🔧 添加学习率调度器（只在完整模型阶段）
        if not attention_training_phase:
            old_lr = optimizer.param_groups[0]['lr']
            scheduler.step(avg_loss)
            new_lr = optimizer.param_groups[0]['lr']
            
            if old_lr != new_lr:
                print(f"🔧 Learning rate reduced: {old_lr:.2e} -> {new_lr:.2e}")

def get_boundaries(sample, conditioning_info):
    """计算token区域边界"""
    boundaries = {}
    pointer = 1  # skip collapse token
    for field in ['hd'] + conditioning_info:
        if field in sample and sample[field].shape[0] > 0:
            L = sample[field].shape[0]
            boundaries[field] = (pointer, pointer + L)
            pointer += L
    return boundaries

def classifier():
    device = get_device()
    global cfg
    cfg = {
        's_in_dim': 21,
        'z_in_dim': 2,
        's_dim': 128,
        'z_dim': 64,
        'N_elayers': 8
    }
    
    model = psiCLM(cfg).to(device)
    
    # 🔧 为不同参数组设置不同的学习率
    attention_params = []
    collapse_params = []
    other_params = []
    
    for name, param in model.named_parameters():
        if 'attn' in name:
            attention_params.append(param)
        elif 'collapse' in name:
            collapse_params.append(param)
        else:
            other_params.append(param)
    
    # 🔧 多组学习率：attention权重用更高学习率
    optimizer = opt.Adam([
        {'params': other_params, 'lr': 5e-5, 'weight_decay': 1e-4},
        {'params': attention_params, 'lr': 1e-4, 'weight_decay': 1e-5},  # 更高学习率
        {'params': collapse_params, 'lr': 2e-4, 'weight_decay': 1e-6},   # 最高学习率
    ])
    
    # 🔧 添加学习率调度器
    scheduler = opt.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.8, patience=20
    )

    # Resume if possible with compatibility handling
    files = [f for f in os.listdir(model_path) if f.startswith('model_epoch_') and not f.endswith('.opt')]
    if files:
        epochs = [int(re.findall(r'\d+', f)[0]) for f in files]
        latest = max(epochs)
        ckpt = os.path.join(model_path, f'model_epoch_{latest}')
        print(f"Resuming from {ckpt}")
        
        # 兼容性加载
        try:
            model.load_state_dict(torch.load(ckpt))
            optimizer.load_state_dict(torch.load(ckpt + '.opt'))
            start = latest
            print(" Successfully loaded checkpoint with all parameters")
        except Exception as e:
            print(f" Checkpoint incompatible: {e}")
            print(" Starting fresh training with new model structure")
            start = 0
    else:
        start = 0

    # # 添加数据分析
    # print("Analyzing training data distribution...")
    # analyze_mask_distribution()

    train(model, optimizer, start, scheduler)

if __name__ == '__main__':
    classifier()
