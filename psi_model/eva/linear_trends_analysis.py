import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import precision_recall_curve, roc_curve, auc, confusion_matrix, classification_report
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

def main():
    print("加载数据...")
    # 读取完整数据
    df = pd.read_csv('all_model_exp.csv')
    
    print(f"数据形状: {df.shape}")
    print(f"数据列: {df.columns.tolist()}")
    
    # 获取模型预测列（score列）
    score_columns = [col for col in df.columns if col.endswith('_score')]
    print(f"找到 {len(score_columns)} 个模型预测列: {score_columns}")
    
    # 1. 线性趋势分析
    print("\n=== 线性趋势分析 ===")
    linear_analysis_results = analyze_linear_trends(df, score_columns)
    
    # 2. 创建线性趋势可视化
    create_linear_trends_plots(df, score_columns)
    
    # 3. 精度分析（基于binary_label）
    print("\n=== 精度分析 ===")
    precision_results = analyze_precision(df, score_columns)
    
    # 4. 创建精度可视化
    create_precision_plots(df, score_columns)
    
    # 5. 保存结果
    save_results(linear_analysis_results, precision_results)
    
    print("\n分析完成！结果已保存到文件中。")

def analyze_linear_trends(df, score_columns):
    """分析log2FoldChange与模型预测之间的线性趋势"""
    results = []
    
    for col in score_columns:
        # 计算相关系数
        pearson_r, pearson_p = stats.pearsonr(df['log2FoldChange'], df[col])
        spearman_r, spearman_p = stats.spearmanr(df['log2FoldChange'], df[col])
        
        # 线性回归分析
        X = df[col].values.reshape(-1, 1)
        y = df['log2FoldChange'].values
        
        lr = LinearRegression()
        lr.fit(X, y)
        
        # 计算R²
        r_squared = lr.score(X, y)
        
        # 预测值
        y_pred = lr.predict(X)
        
        # 计算RMSE
        rmse = np.sqrt(np.mean((y - y_pred)**2))
        
        results.append({
            'model': col.replace('_score', ''),
            'pearson_r': pearson_r,
            'pearson_p': pearson_p,
            'spearman_r': spearman_r,
            'spearman_p': spearman_p,
            'r_squared': r_squared,
            'rmse': rmse,
            'slope': lr.coef_[0],
            'intercept': lr.intercept_
        })
        
        print(f"{col}:")
        print(f"  Pearson相关系数: {pearson_r:.4f} (p={pearson_p:.4f})")
        print(f"  Spearman相关系数: {spearman_r:.4f} (p={spearman_p:.4f})")
        print(f"  R²: {r_squared:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  线性方程: y = {lr.coef_[0]:.4f}x + {lr.intercept_:.4f}")
        print()
    
    return pd.DataFrame(results)

def create_linear_trends_plots(df, score_columns):
    """创建线性趋势可视化图"""
    # 设置图形样式
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 创建散点图和回归线
    n_models = len(score_columns)
    n_cols = 3
    n_rows = (n_models + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, col in enumerate(score_columns):
        row = i // n_cols
        col_idx = i % n_cols
        ax = axes[row, col_idx]
        
        # 散点图
        ax.scatter(df[col], df['log2FoldChange'], alpha=0.6, s=30)
        
        # 计算并绘制回归线
        X = df[col].values.reshape(-1, 1)
        y = df['log2FoldChange'].values
        lr = LinearRegression()
        lr.fit(X, y)
        
        x_range = np.linspace(df[col].min(), df[col].max(), 100)
        y_pred = lr.predict(x_range.reshape(-1, 1))
        ax.plot(x_range, y_pred, 'r-', linewidth=2)
        
        # 计算相关系数
        r, p = stats.pearsonr(df['log2FoldChange'], df[col])
        
        ax.set_xlabel(f'{col}')
        ax.set_ylabel('log2FoldChange')
        ax.set_title(f'{col.replace("_score", "")}\nR = {r:.3f}, p = {p:.3f}')
        ax.grid(True, alpha=0.3)
    
    # 删除多余的子图
    for i in range(n_models, n_rows * n_cols):
        row = i // n_cols
        col_idx = i % n_cols
        fig.delaxes(axes[row, col_idx])
    
    plt.tight_layout()
    plt.savefig('linear_trends_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 创建相关性热图
    plt.figure(figsize=(12, 8))
    
    # 计算所有相关系数
    corr_data = df[['log2FoldChange'] + score_columns].corr()
    
    # 只保留与log2FoldChange的相关性
    log2fc_corr = corr_data['log2FoldChange'].drop('log2FoldChange')
    
    # 创建热图数据
    heatmap_data = pd.DataFrame({
        'Pearson_with_log2FC': log2fc_corr
    })
    
    # 添加Spearman相关系数
    spearman_corr = []
    for col in score_columns:
        r, _ = stats.spearmanr(df['log2FoldChange'], df[col])
        spearman_corr.append(r)
    
    heatmap_data['Spearman_with_log2FC'] = spearman_corr
    heatmap_data.index = [col.replace('_score', '') for col in score_columns]
    
    sns.heatmap(heatmap_data, annot=True, cmap='RdBu_r', center=0, 
                fmt='.3f', cbar_kws={'label': 'Correlation Coefficient'})
    plt.title('模型预测与log2FoldChange的相关性')
    plt.ylabel('模型')
    plt.xlabel('相关性类型')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

def analyze_precision(df, score_columns):
    """分析模型预测的精度"""
    results = []
    
    for col in score_columns:
        # 使用模型score作为预测概率
        y_true = df['binary_label']
        y_scores = df[col]
        
        # 计算ROC曲线
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        # 计算Precision-Recall曲线
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        pr_auc = auc(recall, precision)
        
        # 使用中位数作为阈值进行二分类
        threshold = np.median(y_scores)
        y_pred = (y_scores > threshold).astype(int)
        
        # 计算混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        
        # 计算精度指标
        if len(np.unique(y_pred)) > 1:  # 确保有两个类别
            report = classification_report(y_true, y_pred, output_dict=True)
            
            results.append({
                'model': col.replace('_score', ''),
                'roc_auc': roc_auc,
                'pr_auc': pr_auc,
                'threshold': threshold,
                'accuracy': report['accuracy'],
                'precision_0': report['0']['precision'],
                'recall_0': report['0']['recall'],
                'f1_0': report['0']['f1-score'],
                'precision_1': report['1']['precision'],
                'recall_1': report['1']['recall'],
                'f1_1': report['1']['f1-score'],
                'macro_precision': report['macro avg']['precision'],
                'macro_recall': report['macro avg']['recall'],
                'macro_f1': report['macro avg']['f1-score']
            })
        
        print(f"{col}:")
        print(f"  ROC AUC: {roc_auc:.4f}")
        print(f"  PR AUC: {pr_auc:.4f}")
        print(f"  阈值: {threshold:.4f}")
        if len(np.unique(y_pred)) > 1:
            print(f"  准确率: {report['accuracy']:.4f}")
            print(f"  宏平均F1: {report['macro avg']['f1-score']:.4f}")
        print()
    
    return pd.DataFrame(results)

def create_precision_plots(df, score_columns):
    """创建精度分析可视化图"""
    
    # 1. ROC曲线
    plt.figure(figsize=(12, 8))
    
    for col in score_columns:
        y_true = df['binary_label']
        y_scores = df[col]
        
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, linewidth=2, 
                label=f'{col.replace("_score", "")} (AUC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC曲线比较')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('roc_curves_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Precision-Recall曲线
    plt.figure(figsize=(12, 8))
    
    for col in score_columns:
        y_true = df['binary_label']
        y_scores = df[col]
        
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        pr_auc = auc(recall, precision)
        
        plt.plot(recall, precision, linewidth=2,
                label=f'{col.replace("_score", "")} (AUC = {pr_auc:.3f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall曲线比较')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('pr_curves_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. 模型性能对比柱状图
    results_df = analyze_precision(df, score_columns)
    
    if not results_df.empty:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # ROC AUC
        axes[0,0].bar(results_df['model'], results_df['roc_auc'])
        axes[0,0].set_title('ROC AUC 比较')
        axes[0,0].set_ylabel('ROC AUC')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # PR AUC
        axes[0,1].bar(results_df['model'], results_df['pr_auc'])
        axes[0,1].set_title('PR AUC 比较')
        axes[0,1].set_ylabel('PR AUC')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # Accuracy
        axes[1,0].bar(results_df['model'], results_df['accuracy'])
        axes[1,0].set_title('准确率比较')
        axes[1,0].set_ylabel('Accuracy')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # Macro F1
        axes[1,1].bar(results_df['model'], results_df['macro_f1'])
        axes[1,1].set_title('宏平均F1分数比较')
        axes[1,1].set_ylabel('Macro F1')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('model_performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

def save_results(linear_results, precision_results):
    """保存分析结果"""
    
    # 保存线性趋势分析结果
    linear_results_sorted = linear_results.sort_values('r_squared', ascending=False)
    linear_results_sorted.to_csv('linear_trends_results.csv', index=False)
    
    print("线性趋势分析结果 (按R²排序):")
    print(linear_results_sorted[['model', 'pearson_r', 'spearman_r', 'r_squared', 'rmse']].to_string(index=False))
    
    # 保存精度分析结果
    if not precision_results.empty:
        precision_results_sorted = precision_results.sort_values('roc_auc', ascending=False)
        precision_results_sorted.to_csv('precision_analysis_results.csv', index=False)
        
        print("\n精度分析结果 (按ROC AUC排序):")
        print(precision_results_sorted[['model', 'roc_auc', 'pr_auc', 'accuracy', 'macro_f1']].to_string(index=False))
    
    # 创建综合结果总结
    summary = []
    
    for _, row in linear_results.iterrows():
        model_name = row['model']
        
        # 从精度结果中找到对应模型
        precision_row = precision_results[precision_results['model'] == model_name]
        
        summary_item = {
            'model': model_name,
            'pearson_r': row['pearson_r'],
            'spearman_r': row['spearman_r'],
            'r_squared': row['r_squared'],
            'rmse': row['rmse']
        }
        
        if not precision_row.empty:
            summary_item.update({
                'roc_auc': precision_row.iloc[0]['roc_auc'],
                'pr_auc': precision_row.iloc[0]['pr_auc'],
                'accuracy': precision_row.iloc[0]['accuracy'],
                'macro_f1': precision_row.iloc[0]['macro_f1']
            })
        
        summary.append(summary_item)
    
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv('comprehensive_model_analysis.csv', index=False)
    
    print("\n综合分析结果已保存到 'comprehensive_model_analysis.csv'")

if __name__ == "__main__":
    main() 