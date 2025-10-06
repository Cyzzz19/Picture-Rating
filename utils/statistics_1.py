import json
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

def analyze_json_value_distribution(json_file_path):
    """
    分析JSON键值的分布情况（值范围在0-10）
    
    参数:
    json_file_path: JSON文件路径
    """
    try:
        # 读取JSON文件
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 提取所有值
        values = list(data.values())
        
        print("=" * 50)
        print("JSON键值分布分析报告")
        print("=" * 50)
        
        # 基本统计信息
        print(f"总键值对数量: {len(values)}")
        print(f"值范围: {min(values):.1f} - {max(values):.1f}")
        print(f"平均值: {np.mean(values):.2f}")
        print(f"中位数: {np.median(values):.2f}")
        print(f"标准差: {np.std(values):.2f}")
        
        # 统计0-10范围内的分布
        value_counts = Counter()
        
        # 将值按0-10的整数进行分组统计
        for value in values:
            # 将值四舍五入到最近的整数
            rounded_value = round(value)
            # 限制在0-10范围内
            bucket = max(0, min(10, rounded_value))
            value_counts[bucket] += 1
        
        print("\n值分布统计 (四舍五入到整数):")
        print("-" * 30)
        for i in range(11):
            count = value_counts[i]
            percentage = (count / len(values)) * 100
            print(f"值 {i}: {count:4d} 个 ({percentage:5.1f}%)")
        
        # 可视化
        create_visualizations(values, value_counts)
        
    except Exception as e:
        print(f"错误：无法处理JSON文件 - {e}")

def create_visualizations(values, value_counts):
    """
    创建可视化图表
    """
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建子图
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('JSON键值分布分析', fontsize=16, fontweight='bold')
    
    # 1. 条形图 - 整数分布
    buckets = list(range(11))
    counts = [value_counts[i] for i in buckets]
    
    bars = ax1.bar(buckets, counts, color='skyblue', edgecolor='black', alpha=0.7)
    ax1.set_xlabel('值')
    ax1.set_ylabel('数量')
    ax1.set_title('值分布条形图 (四舍五入到整数)')
    ax1.set_xticks(buckets)
    
    # 在条形上显示数量
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + max(counts)*0.01,
                f'{count}', ha='center', va='bottom', fontsize=9)
    
    # 2. 饼图 - 分布比例
    labels = [f'值 {i}' for i in buckets]
    sizes = counts
    # 只显示数量大于0的部分
    non_zero_indices = [i for i, count in enumerate(counts) if count > 0]
    non_zero_labels = [labels[i] for i in non_zero_indices]
    non_zero_sizes = [sizes[i] for i in non_zero_indices]
    
    wedges, texts, autotexts = ax2.pie(non_zero_sizes, labels=non_zero_labels, autopct='%1.1f%%',
                                      startangle=90, colors=plt.cm.Set3(np.linspace(0, 1, len(non_zero_sizes))))
    ax2.set_title('值分布饼图')
    
    # 3. 直方图 - 原始值分布
    ax3.hist(values, bins=20, color='lightgreen', edgecolor='black', alpha=0.7)
    ax3.set_xlabel('值')
    ax3.set_ylabel('频率')
    ax3.set_title('原始值分布直方图')
    ax3.grid(True, alpha=0.3)
    
    # 4. 箱线图 - 统计摘要
    ax4.boxplot(values, vert=True, patch_artist=True,
               boxprops=dict(facecolor='lightcoral', color='black'),
               medianprops=dict(color='red'))
    ax4.set_ylabel('值')
    ax4.set_title('值分布箱线图')
    ax4.set_xticks([1])
    ax4.set_xticklabels(['值分布'])
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 打印详细统计
    print("\n详细统计信息:")
    print("-" * 30)
    for percentile in [25, 50, 75, 90, 95]:
        p_value = np.percentile(values, percentile)
        print(f"{percentile}% 的值 <= {p_value:.2f}")

def main():
    """主函数"""
    print("JSON键值分布分析工具")
    print("=" * 40)
    
    json_file_path = input("请输入JSON文件路径: ").strip()
    
    if not json_file_path:
        print("错误：请输入有效的文件路径")
        return
    
    if not json_file_path.endswith('.json'):
        json_file_path += '.json'
    
    try:
        analyze_json_value_distribution(json_file_path)
    except FileNotFoundError:
        print(f"错误：找不到文件 {json_file_path}")
    except json.JSONDecodeError:
        print("错误：JSON文件格式不正确")

# 简化版本，直接指定文件路径
def quick_analysis():
    """快速分析版本"""
    json_file_path = "D:\Picture Rating\scores.json"  # 替换为您的JSON文件路径
    analyze_json_value_distribution(json_file_path)

if __name__ == "__main__":
    # 使用交互式版本
    quick_analysis()
    
    # 或者取消注释下面这行使用快速版本（需要先修改路径）
    # quick_analysis()