"""
GRPO模块冒泡测试
快速验证GRPO损失计算是否正常工作
"""

import warnings

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message="The pynvml package is deprecated.*",
)
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from groot.vla.grpo_simple import SimpleGRPOLoss, GRPORewardShaper

def test_grpo_loss():
    """测试GRPO损失计算"""
    print("=" * 50)
    print("测试1: GRPO损失计算")
    print("=" * 50)
    
    # 创建GRPO损失函数
    grpo_loss_fn = SimpleGRPOLoss(reward_scale=1.0, advantage_normalize=True)
    
    # 模拟数据
    batch_size = 4
    log_probs = torch.randn(batch_size)  # 随机对数概率
    rewards = torch.randn(batch_size)    # 随机奖励
    
    print(f"Log probs: {log_probs}")
    print(f"Rewards: {rewards}")
    
    # 计算损失
    loss = grpo_loss_fn(log_probs, rewards)
    
    print(f"GRPO Loss: {loss.item():.4f}")
    print("✓ GRPO损失计算成功\n")
    
    return loss

def test_reward_shaper():
    """测试奖励塑形器"""
    print("=" * 50)
    print("测试2: 奖励塑形器")
    print("=" * 50)
    
    # 创建奖励塑形器
    reward_shaper = GRPORewardShaper(reward_scale=0.01)
    
    # 模拟批次数据
    batch_size = 4
    seq_len = 10
    action_dim = 8
    
    batch = {
        'action': torch.randn(batch_size, 1, seq_len, action_dim),
        'action_mask': torch.ones(batch_size, 1, seq_len, action_dim, dtype=torch.bool),
    }
    
    print(f"Batch action shape: {batch['action'].shape}")
    
    # 计算奖励
    rewards = reward_shaper.compute_batch_rewards(batch)
    
    print(f"Rewards shape: {rewards.shape}")
    print(f"Rewards: {rewards}")
    print(f"Mean reward: {rewards.mean().item():.4f}")
    reward_std = rewards.std().item() if rewards.numel() > 1 else 0.0
    print(f"Std reward: {reward_std:.4f}")
    print("✓ 奖励塑形器计算成功\n")
    
    return rewards

def test_integration():
    """测试BC+GRPO集成"""
    print("=" * 50)
    print("测试3: BC+GRPO集成")
    print("=" * 50)
    
    # 模拟BC损失
    bc_loss = torch.tensor(0.5)
    print(f"BC Loss: {bc_loss.item():.4f}")
    
    # 创建GRPO组件
    grpo_loss_fn = SimpleGRPOLoss()
    reward_shaper = GRPORewardShaper(reward_scale=0.01)
    
    # 模拟数据
    batch_size = 4
    seq_len = 10
    action_dim = 8
    
    batch = {
        'action': torch.randn(batch_size, 1, seq_len, action_dim),
        'action_mask': torch.ones(batch_size, 1, seq_len, action_dim, dtype=torch.bool),
    }
    
    # 计算奖励
    rewards = reward_shaper.compute_batch_rewards(batch)
    
    # 模拟对数概率
    log_probs = torch.randn(batch_size)
    
    # 计算GRPO损失
    grpo_loss = grpo_loss_fn(log_probs, rewards)
    print(f"GRPO Loss: {grpo_loss.item():.4f}")
    
    # 总损失
    grpo_weight = 0.1
    total_loss = bc_loss + grpo_weight * grpo_loss
    
    print(f"Total Loss: {total_loss.item():.4f}")
    print(f"  = BC Loss ({bc_loss.item():.4f}) + {grpo_weight} * GRPO Loss ({grpo_loss.item():.4f})")
    print("✓ BC+GRPO集成测试成功\n")
    
    return total_loss

def main():
    print("\n" + "=" * 50)
    print("GRPO模块冒泡测试")
    print("=" * 50 + "\n")
    
    try:
        # 测试1: GRPO损失
        test_grpo_loss()
        
        # 测试2: 奖励塑形器
        test_reward_shaper()
        
        # 测试3: BC+GRPO集成
        test_integration()
        
        print("=" * 50)
        print("✓ 所有测试通过！")
        print("=" * 50)
        
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
