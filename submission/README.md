# Stroke SHAP 代码作业和英文报告

本文件夹按作业截图中的六项任务整理，英文报告沿用提供的 Word 模板。实验已实际运行，结果和图表均来自随附 CSV。

## 先看这些文件

- `Stroke_SHAP_Report.docx`：英文报告，包含实验方法、结果、局部和全局解释、模型比较、参考文献和复现说明。
- `Stroke_SHAP_Audience_Slides.pptx`：恰好三页英文幻灯片，分别面向数据科学家、医生和患者，带英文演讲备注。
- `Stroke_SHAP_Assignment.ipynb`：已执行的配套 Notebook，按六项任务展示结果，完整实现复用随附 Python 文件。
- `code/run_assignment.py`：完整训练与解释流程，运行后重新生成实验结果。
- `code/00_data_exploration.py`：执行原始教学 EDA，并补充数据质量检查和汇总图。
- `code/verify_outputs.py`：重新加载模型，独立检查预测、指标、数据划分和 SHAP 加和结果。

报告已按老师原始模板逐项复核，9 张表保留原表头、固定项目名称及顺序。封面的英文姓名、学号、课程名称和老师姓名已保留本人填写的信息；Submission date（提交日期）按本人确认填写为 12 November 2026，提交检查项已全部勾选。

## 六项任务对应内容

1. EDA：原始教学脚本已执行，12 张原始直方图、控制台记录及补充分析位于 `outputs/eda/`。
2. SHAP explainer：随机森林和 Tree SHAP，解释 `stroke = 1` 的概率尺度输出；参考为经过类别加权的训练路径。
3. 局部 force plot：`outputs/figures/tree_force.png` 和可交互的 `tree_force.html`。
4. 全局 summary plot：`outputs/figures/tree_summary.png`，解释样本包含 120 条测试记录。
5. 三页幻灯片：单独的 PPTX 文件，使用报告中相同模型、案例和数值。
6. 另一黑箱模型：多层神经网络和 Kernel SHAP，含局部、全局解释与背景敏感性检查。

## 实验设置和主要结果

- 原始数据 5,110 行、12 列；249 条中风记录，10 个实际预测变量。
- 分层划分：训练 3,066，验证 1,022，测试 1,022；种子 2021。
- 填补、编码、缩放和重采样仅使用训练集；分类阈值仅使用验证集 F1 选择。
- 随机森林：测试 ROC AUC 0.8646、average precision 0.2217、召回率 0.52、F1 0.2842，阈值 0.52。
- 神经网络：测试 ROC AUC 0.7032、average precision 0.1299、召回率 0.24、F1 0.1371，阈值 0.04。
- 选定的测试案例为 CSV 零基行号 18，两种模型都漏判；报告如实分析这一错误。
- 神经网络在 600 轮达到迭代上限，并保留数值运算提示。所有保存的权重、预测、解释值均通过有限值和加和核验。报告已讨论这些训练限制。
- 输出是对“已记录中风状态”的模型评分，没有被验证为未来中风风险。SHAP 解释模型行为，不代表因果关系。

## 重新运行

建议使用 Python 3.12，并在本文件夹建立独立环境。以下命令在 `submission` 文件夹中执行：

```bash
python -m venv .venv
```

macOS 或 Linux 激活：

```bash
source .venv/bin/activate
```

Windows 激活：

```powershell
.venv\Scripts\Activate.ps1
```

安装依赖并运行：

```bash
python -m pip install -r requirements.txt
python code/run_assignment.py
python code/verify_outputs.py
```

只运行 Task 1：

```bash
python code/00_data_exploration.py
```

Notebook 已保存运行结果，打开即可查看。如需在已有 Jupyter 环境重新执行，使用安装本项目依赖的 Python 内核；额外的内核依赖在 `requirements-notebook.txt` 中。Notebook 默认读取已保存的结果，将第一段代码的 `REGENERATE` 改为 `True` 可重新训练。不同平台的运行时间和数值细节可能略有差异。

训练脚本重新生成数据、模型与图表；Word 和 PPT 是随本次已验证实验结果整理的成品，不会被训练脚本自动改写。

## 结果目录

- `outputs/metrics.json`：验证和测试指标，以及阈值 0.5 的补充结果。
- `outputs/run_details.json`：版本、参数、种子、案例、SHAP 验证结果和训练提示。
- `outputs/tables/`：划分归属、逐条测试预测、阈值选择、局部贡献和全局排名。
- `outputs/explanations/shap_values.npz`：完整 SHAP 数组、背景和解释样本索引、特征顺序和选定案例位置。
- `outputs/explanations/category_codes.json`：Kernel SHAP 使用的可逆类别映射。该编码仅作解释器传输，模型仍使用 one-hot 编码。
- `outputs/models/`：两个已拟合的完整预测流程；重新训练可再生。
- `outputs/figures/`：force、summary、waterfall、混淆矩阵和性能曲线。
- `outputs/execution.log`、`outputs/verification.log`：实际执行与独立核验记录。
- `environment-lock.txt`：本次实际环境的完整依赖版本；`requirements.txt` 只列主要运行依赖。
- `original/`：老师提供的示例代码和要求截图，供溯源核对。

## 代码来源和协助说明

原始教学代码来源为 https://github.com/deepfindr/xai-series ，提交版本 `91542416a4924182685104a4192dc037b562ddec`。数据来源见原始 README 和英文报告参考文献。

原始教学文件保留；本作业新增流程修正了 BMI 填零、非分层划分、解释与绘图样本数不一致，以及新旧 SHAP 输出维度的兼容问题。

OpenAI Codex 协助了代码、实验、图表、报告和幻灯片。请本人核对内容、理解运行过程，并按课程要求处理使用协助的说明。
