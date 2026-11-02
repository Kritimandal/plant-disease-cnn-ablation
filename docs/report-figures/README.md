# Report figures

All figures from the seminar report and presentation, kept so the reported numbers can be checked against them.

## Final study (40-epoch ablation run, seed 42)

| Figure | Shows |
|---|---|
| [fig1_workflow.png](fig1_workflow.png) | Experiment workflow, from leakage audit to accuracy and efficiency measurement |
| [fig2_architecture.png](fig2_architecture.png) | Architectures of arms A0–A4; shaded blocks mark the change from the arm above |
| [class_distribution.png](class_distribution.png) | Number of training images per class (38 classes) |
| [out1_splits.png](out1_splits.png) | Output of `make_splits.py`: files scanned, leakage check and split sizes |
| [fig3_macro_f1.png](fig3_macro_f1.png) | Test macro F1 for every arm |
| [out2_analysis.png](out2_analysis.png) | Output of `analyze_ablation.py`: comparison table and McNemar tests against A0 |
| [ablation_tradeoff.png](ablation_tradeoff.png) | Macro F1 against parameters and against CPU latency |
| [fig4_accuracy_vs_size.png](fig4_accuracy_vs_size.png) | Macro F1 against parameters and against saved model size |
| [fig5_flops_vs_latency.png](fig5_flops_vs_latency.png) | Operation count against measured CPU latency |
| [fig6_confusion_a4.png](fig6_confusion_a4.png) | Confusion matrix for A4 on the test split |
| [fig4_confusion_a4.png](fig4_confusion_a4.png) | Confusion matrix for A4 with the count in every non-zero cell |
| [fig7_validation_curves.png](fig7_validation_curves.png) | Validation accuracy by epoch for all six arms |
| [code1_baseline.png](code1_baseline.png) | Code: A0 baseline constructor |
| [code2_inverted_residual.png](code2_inverted_residual.png) | Code: inverted residual block and A4 |
| [code3_grouped_split.png](code3_grouped_split.png) | Code: source-grouped splitting |
| [code4_protocol.png](code4_protocol.png) | Code: training and evaluation protocol |

## Earlier run (superseded)

These come from the first experiment, which compared only two models trained for different numbers of epochs on the dataset's original, leaking split. They are kept as a record of why the study was redesigned and are **not** part of the final results.

| Figure | Shows |
|---|---|
| [fig0_problem.png](fig0_problem.png) | Why the first comparison could not be trusted: where the baseline's parameters sit, the differences between the two models, and the leaking validation folder |
| [pipeline_diagram.png](pipeline_diagram.png) | Pipeline of the first experiment |
| [custom_cnn_curves.png](custom_cnn_curves.png) | Accuracy and loss of the first custom CNN run (4 epochs) |
| [mobilenet_curves.png](mobilenet_curves.png) | Accuracy and loss of the first MobileNetV2 run (10 epochs) |
