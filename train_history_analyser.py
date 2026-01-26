import matplotlib.pyplot as plt
import pandas as pd

class TrainHistoryAnalyser:
    def __init__(self, model_name, history):
        self._history = history
        self._model_name = model_name

    @property
    def history(self):
        return self._history
    @history.setter
    def history(self, new_hist):
        self._history = new_hist

    def inspect_history(self):
        """详细检查训练历史"""
        
        # 检查是否有 history 属性
        if hasattr(self._history, 'history'):
            history_dict = self._history.history
            
            print("=" * 60)
            print("训练历史详情:")
            print("=" * 60)
            
            # 列出所有可用的指标
            print("\n可用的指标:")
            for key in history_dict.keys():
                print(f"  - {key}")
            
            # 查看每个指标的最后一个值
            print("\n最终指标值:")
            for key, values in history_dict.items():
                if len(values) > 0:
                    print(f"  {key}: {values[-1]:.4f}")
                else:
                    print(f"  {key}: 无数据")
            
            # 查看每个指标的长度（epoch数）
            print("\n训练周期数:")
            for key, values in history_dict.items():
                print(f"  {key}: {len(values)} 个值")
            
            return history_dict
        else:
            print("History 对象没有 history 属性")
            return None


    def plot_training_history(self):
        """绘制训练历史图表"""
        
        if not hasattr(self.history, 'history'):
            print("无法绘制：History 对象没有 history 属性")
            return
        
        history_dict = self.history.history
        epochs = range(1, len(history_dict['loss']) + 1)
        
        # 创建图表
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        self._plot_loss_curve(axes, history_dict, epochs)
        self._plot_accuracy_curve(axes, history_dict, epochs)

        plt.tight_layout()
        plt.show()

    def _plot_loss_curve(self, axes, history_dict, epochs):    
        # 1. 绘制损失曲线
        if 'loss' in history_dict and 'val_loss' in history_dict:
            axes[0].plot(epochs, history_dict['loss'], 'b-', label='Train Loss')
            axes[0].plot(epochs, history_dict['val_loss'], 'r-', label='Valuation Loss')
            axes[0].set_title('Train and Valuation Loss')
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Loss')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
        elif 'loss' in history_dict:
            axes[0].plot(epochs, history_dict['loss'], 'b-', label='Train Loss')
            axes[0].set_title('Train Loss')
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Loss')
            axes[0].grid(True, alpha=0.3)

    def _plot_accuracy_curve(self, axes, history_dict, epochs):    
        # 2. 绘制准确率曲线
        accuracy_key = None
        val_accuracy_key = None
        
        # 查找准确率相关的键
        for key in history_dict.keys():
            if 'accuracy' in key.lower() and 'val' not in key:
                accuracy_key = key
            elif 'val_accuracy' in key.lower() or ('val' in key and 'accuracy' in key.lower()):
                val_accuracy_key = key
        
        if accuracy_key:
            axes[1].plot(epochs, history_dict[accuracy_key], 'b-', label='Train Accuracy')
            if val_accuracy_key:
                axes[1].plot(epochs, history_dict[val_accuracy_key], 'r-', label='Valuation Accuracy')
            axes[1].set_title('Train and Valuation Accuracy')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Accuracy')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        
    def _plot_other_curve(self, history_dict, epochs):
        # 3. 绘制其他指标（如果有）
        other_metrics = []
        for key in history_dict.keys():
            if key not in ['loss', 'val_loss', 'accuracy', 'val_accuracy', 'lr']:
                other_metrics.append(key)
        
        if other_metrics:
            fig, axes = plt.subplots(1, len(other_metrics), figsize=(5*len(other_metrics), 4))
            
            if len(other_metrics) == 1:
                axes = [axes]  # 确保 axes 是列表
            
            for i, metric in enumerate(other_metrics):
                axes[i].plot(epochs, history_dict[metric], 'g-')
                axes[i].set_title(f'{metric}')
                axes[i].set_xlabel('Epoch')
                axes[i].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()

    def interactive_history_analysis(self):
        """交互式分析训练历史"""
        
        if not hasattr(self.history, 'history'):
            return
        
        history_dict = self.history.history
        metrics = list(history_dict.keys())
        
        print("可分析的指标:")
        for i, metric in enumerate(metrics):
            print(f"{i+1}. {metric}")
        
        # 让用户选择要分析的指标
        while True:
            try:
                choice = input("\n输入要分析的指标编号（输入 q 退出）: ")
                if choice.lower() == 'q':
                    break
                
                idx = int(choice) - 1
                if 0 <= idx < len(metrics):
                    metric = metrics[idx]
                    values = history_dict[metric]
                    
                    # 显示统计信息
                    print(f"\n{metric} 分析:")
                    print(f"  最小值: {min(values):.6f} (epoch {values.index(min(values))+1})")
                    print(f"  最大值: {max(values):.6f} (epoch {values.index(max(values))+1})")
                    print(f"  平均值: {sum(values)/len(values):.6f}")
                    print(f"  最终值: {values[-1]:.6f}")
                    print(f"  变化趋势: {'上升' if values[-1] > values[0] else '下降'}")
                    
                    # 绘制单个指标
                    plt.figure(figsize=(10, 4))
                    plt.plot(range(1, len(values)+1), values, 'b-', linewidth=2)
                    plt.title(f'{metric} 变化曲线')
                    plt.xlabel('Epoch')
                    plt.ylabel(metric)
                    plt.grid(True, alpha=0.3)
                    
                    # 标记最佳值
                    if 'loss' in metric:
                        best_idx = values.index(min(values))
                        best_value = min(values)
                        plt.plot(best_idx+1, best_value, 'ro', markersize=10, label=f'最佳值: {best_value:.4f}')
                    elif 'accuracy' in metric.lower():
                        best_idx = values.index(max(values))
                        best_value = max(values)
                        plt.plot(best_idx+1, best_value, 'ro', markersize=10, label=f'最佳值: {best_value:.4f}')
                    
                    plt.legend()
                    plt.show()
                    
                else:
                    print("无效的选择")
            except ValueError:
                print("请输入数字或 'q' 退出")

    def analyze_training_performance(self):
        """深入分析训练性能"""
        
        if not hasattr(self.history, 'history'):
            return None
        
        history_dict = self.history.history
        analysis = {}
        
        print("=" * 60)
        print("训练性能深度分析")
        print("=" * 60)
        
        # 分析过拟合/欠拟合
        if 'loss' in history_dict and 'val_loss' in history_dict:
            train_loss = history_dict['loss']
            val_loss = history_dict['val_loss']
            
            # 计算过拟合指标
            final_overfit = val_loss[-1] - train_loss[-1]
            max_overfit = max(val_loss) - min(train_loss)
            
            analysis['overfitting'] = {
                'final_difference': final_overfit,
                'max_difference': max_overfit,
                'final_train_loss': train_loss[-1],
                'final_val_loss': val_loss[-1]
            }
            
            print(f"\n过拟合分析:")
            print(f"  最终训练损失: {train_loss[-1]:.4f}")
            print(f"  最终验证损失: {val_loss[-1]:.4f}")
            print(f"  最终差值: {final_overfit:.4f}")
            
            if final_overfit > 0.1:
                print(f"  ⚠️ 警告：可能存在过拟合（验证损失明显高于训练损失）")
            elif final_overfit < -0.1:
                print(f"  ⚠️ 警告：可能存在欠拟合")
            else:
                print(f"  ✓ 训练和验证损失匹配良好")
        
        # 分析收敛性
        if 'loss' in history_dict:
            loss_values = history_dict['loss']
            
            # 计算收敛速度
            if len(loss_values) >= 10:
                early_avg = sum(loss_values[:5]) / 5
                late_avg = sum(loss_values[-5:]) / 5
                improvement = early_avg - late_avg
                
                analysis['convergence'] = {
                    'early_avg_loss': early_avg,
                    'late_avg_loss': late_avg,
                    'improvement': improvement,
                    'improvement_percent': (improvement / early_avg) * 100
                }
                
                print(f"\n收敛分析:")
                print(f"  前5个epoch平均损失: {early_avg:.4f}")
                print(f"  最后5个epoch平均损失: {late_avg:.4f}")
                print(f"  改进幅度: {improvement:.4f} ({improvement/early_avg*100:.1f}%)")
        
        # 分析准确率（如果有）
        accuracy_key = None
        val_accuracy_key = None
        
        for key in history_dict.keys():
            if 'accuracy' in key.lower() and 'val' not in key:
                accuracy_key = key
            elif 'val_accuracy' in key.lower() or ('val' in key and 'accuracy' in key.lower()):
                val_accuracy_key = key
        
        if accuracy_key:
            accuracy_values = history_dict[accuracy_key]
            analysis['accuracy'] = {
                'final': accuracy_values[-1],
                'max': max(accuracy_values),
                'improvement': accuracy_values[-1] - accuracy_values[0]
            }
            
            print(f"\n训练准确率分析:")
            print(f"  初始准确率: {accuracy_values[0]:.4f}")
            print(f"  最终准确率: {accuracy_values[-1]:.4f}")
            print(f"  最高准确率: {max(accuracy_values):.4f}")
            print(f"  提升幅度: {accuracy_values[-1] - accuracy_values[0]:.4f}")
        
        if val_accuracy_key:
            val_accuracy_values = history_dict[val_accuracy_key]
            analysis['val_accuracy'] = {
                'final': val_accuracy_values[-1],
                'max': max(val_accuracy_values),
                'improvement': val_accuracy_values[-1] - val_accuracy_values[0]
            }
            
            print(f"\n验证准确率分析:")
            print(f"  初始验证准确率: {val_accuracy_values[0]:.4f}")
            print(f"  最终验证准确率: {val_accuracy_values[-1]:.4f}")
            print(f"  最高验证准确率: {max(val_accuracy_values):.4f}")
        
        # 查找最佳epoch
        if 'val_loss' in history_dict:
            best_epoch = history_dict['val_loss'].index(min(history_dict['val_loss'])) + 1
            analysis['best_epoch'] = best_epoch
            print(f"\n最佳epoch: {best_epoch}")
            print(f"  该epoch的验证损失: {min(history_dict['val_loss']):.4f}")
        
        return analysis

    # 使用示例：比较两个模型的训练历史
    # history1 = model1.fit(...)
    # history2 = model2.fit(...)
    # comparison = compare_training_histories([history1, history2], ['模型A', '模型B'])
    def compare_training_histories(self, histories, model_names=None):
        """比较多个训练历史"""
        
        if model_names is None:
            model_names = [f'model{i+1}' for i in range(len(histories))]
        
        # 确保每个历史都有数据
        valid_histories = []
        valid_names = []
        for i, history in enumerate(histories):
            if hasattr(history, 'history'):
                valid_histories.append(history.history)
                valid_names.append(model_names[i])
            else:
                print(f"警告: {model_names[i]} 没有有效的训练历史")
        
        if len(valid_histories) < 2:
            print("至少需要两个有效的训练历史进行比较")
            return
        
        # 绘制比较图
        metrics = set()
        for history in valid_histories:
            metrics.update(history.keys())
        
        # 排除学习率等非主要指标
        main_metrics = [m for m in metrics if m not in ['lr']]
        
        # 创建图表
        n_metrics = len(main_metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 4))
        
        if n_metrics == 1:
            axes = [axes]
        
        for idx, metric in enumerate(main_metrics):
            ax = axes[idx]
            
            for i, (history, name) in enumerate(zip(valid_histories, valid_names)):
                if metric in history:
                    values = history[metric]
                    epochs = range(1, len(values) + 1)
                    ax.plot(epochs, values, label=name, linewidth=2, alpha=0.8)
            
            ax.set_title(f'{metric} Compare')
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # 创建比较表格
        comparison_data = []
        
        for name, history in zip(valid_names, valid_histories):
            row = {'Model': name}
            
            for metric in main_metrics:
                if metric in history:
                    values = history[metric]
                    row[f'{metric}_Start'] = values[0]
                    row[f'{metric}_End'] = values[-1]
                    row[f'{metric}_Best'] = min(values) if 'loss' in metric else max(values)
            
            comparison_data.append(row)
        
        df_comparison = pd.DataFrame(comparison_data)
        
        print("\n模型性能比较:")
        print("=" * 80)
        print(df_comparison.to_string(index=False))
        
        return df_comparison

    def get_best_epoch(self, metric='val_loss', mode='min'):
        """获取最佳epoch
        
        参数:
            history: History对象
            metric: 要评估的指标，如 'val_loss', 'val_accuracy'
            mode: 'min' 或 'max'，表示越小越好还是越大越好
        """
        
        if not hasattr(self.history, 'history'):
            return None
        
        history_dict = self.history.history
        
        if metric not in history_dict:
            print(f"指标 '{metric}' 不存在")
            available = list(history_dict.keys())
            print(f"可用指标: {available}")
            return None
        
        values = history_dict[metric]
        
        if mode == 'min':
            best_value = min(values)
            best_epoch = values.index(best_value)
        elif mode == 'max':
            best_value = max(values)
            best_epoch = values.index(best_value)
        else:
            print(f"无效的模式: {mode}，应为 'min' 或 'max'")
            return None
        
        return {
            'epoch': best_epoch + 1,  # epoch从1开始计数
            'value': best_value,
            'all_values': values
        }

    def find_early_stopping_point(self, patience=5, metric='val_loss'):
        """找到早期停止的最佳点"""
        
        if not hasattr(self.history, 'history') or metric not in self.history.history:
            return None
        
        values = self.history.history[metric]
        
        if len(values) <= patience:
            return {'epoch': len(values), 'value': values[-1]}
        
        best_value = values[0]
        best_epoch = 0
        
        for i in range(1, len(values)):
            if values[i] < best_value:  # 假设是损失，越小越好
                best_value = values[i]
                best_epoch = i
            elif i - best_epoch >= patience:
                return {'epoch': best_epoch + 1, 'value': best_value}
        
        return {'epoch': best_epoch + 1, 'value': best_value}


if __name__ == '__main__':
    import sys
    # 使用函数
    analyser = TrainHistoryAnalyser(sys.argv[0], sys.argv[1])
    if sys.argv[0] == '-i':
        print("Interactive History Analysis ...")
        analyser.interactive_history_analysis()
        sys.exit(0)
    # parameters: model name, history object
    print(f"Parameters: {sys.argv[0]}, {sys.argv[1]}")
    analyser.inspect_history()
    analyser.plot_training_history()
    performance_analysis = analyser.analyze_training_performance()
    # 使用示例
    best_epoch_info = analyser.get_best_epoch(metric='val_accuracy', mode='max')
    if best_epoch_info:
        print(f"最佳epoch: {best_epoch_info['epoch']}")
        print(f"最佳准确率: {best_epoch_info['value']:.4f}")

    early_stop_info = analyser.find_early_stopping_point(patience=10, metric='val_loss')
    if early_stop_info:
        print(f"早期停止点: epoch {early_stop_info['epoch']}")
        print(f"损失值: {early_stop_info['value']:.4f}")
