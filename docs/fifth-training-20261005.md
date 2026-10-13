# 第五次训练：退款执行／补救结果承诺

## 状态与启动

**最新核查：第五次已完成**，2026-10-05 23:34:52（北京时间）结束，5轮13455步、best epoch5、全dev loss1.1029298457。实际v3.1数据哈希与冻结一致。已加载新best.pt生成同135条并评分；结果及副作用见[第五次评估](fifth-evaluation-20261005.md)。不要再次启动同任务作为“继续训练”。下面首次核查段保留历史时点，不代表当前仍运行。

准备和冻结已完成。用户已在VS Code启动第五次，运行目录`runs/20261005T132346-full-ebf624ab40734b858b46553b4da37537`。首次核查state=running、phase=baseline_dev、0步0轮，正在计算初始dev loss，不是卡在训练batch；experiment.json确认最终v3.1清单及实际train哈希匹配。这里记录的是当时状态，不表示完成。

默认构建任务为“第五次训练（v3退款执行承诺整句改写，5轮）”。需要新实验时在VS Code按 **Ctrl+Shift+B** 启动，无需输入命令；或“终端 → 运行任务”选择同名任务。当前已运行，不要再按一次启动。先核验冻结清单，成功后训练，输出在“终端”，不是“调试控制台”。完整冻结回放可能需几十秒，请等待。

不要同时运行其他全量任务。运行中终止会保留日志和已有checkpoint，但本项目尚未实现完整断点续训，重启该任务会新建运行。训练完成应显示Finished及best.pt路径；断电时先检查运行status.json，不靠最后一条batch输出判断完成。

## 唯一因素

第四次冻结v2为基线，只改train.response中确认的退款执行/补救/付款结果断言，改为核查和协助申请的整句。未改变问题、ID、顺序、其他字段或实体占位符。正常努力意愿、可能性、引导申请保留；查询准确性、隐私和订阅不是本次新增范围。退款语境的cancel_order3条也在本因素内，并非新增取消订单干预。

全train扫描产生2901个唯一候选句（3074处），词法保守筛选603句逐句AI审阅，再复核79个执行/补救筛除句（包括we'll缩写），共682份明确决策。批准522个精确整句规则，其余有拒绝理由。词法筛除的其他2219句没有被宣称全部人工审阅；这不是全训练语义风险清零。必要模糊语境另核查原响应。批准与拒绝见`outputs/target-revision-v3-review-20261005/approval-freeze-ready.json`。

修改495条/21528条（529处）：check_refund_policy431、get_refund57、track_refund4、cancel_order3。清单含每个ID完整前后response、每处原句/替换句及偏移，不是简单关键词替换。

预检发现初始v3有6条新重复核查提示，启动前按未授权扣款/重复扣款语境细化10个已批准替换句，冻结v3.1；批准片段与因素不变。初始v3及中间审核草案保留追溯，**训练只使用最终v3.1**。最后10句在最终rules.json和manifest.replacement_refinement中记录。保留标题、条件、请求信息、实体；混合句中的查交易、确认邮件等非本因素内容有专门保留决策，不顺带修正。

## 冻结依据

- 最终清单：`data/derived/refund-execution-v3.1/manifest.json`，SHA256 `5c281ecb802be3208bfcf0e0fa7d1a11f3a41a6ee79380e876f7f3c593f4729c`。
- 最终train SHA256：`4d1ed247e99b7b9aa5914be35ea9e6bcb8d4a8f70f1188808cda5b8bd07fbe73`。
- 同目录`changes.jsonl`、`rules.json`、`train.jsonl`均绑定哈希；审阅程序、审核草案及最终批准、原始train/v2、tokenizer、dev、模型、生成和第五次配置依赖已记录。运行前完整精确回放并核验。
- 参数配置：`configs/training-fifth-final.json`。与第四次逐字段比较，除train_revision外完全一致：5轮、batch8、lr0.0003、weight_decay0.01、clip1、seed42、device auto、CPU2线程、2691批/轮。重新初始化，不加载第四次权重续训。
- 训练引擎仍是原`training.py`。最终入口`training_fifth_final.py`仅在当前进程扩展revision分派，结束后还原；不改被v1/v2绑定的旧router、Dataset或历史配置。常规旧入口不认识新revision，请使用第五次VS Code任务。

## 验证与后续

实体多重集与所有非response字段不变；529处批准片段回放匹配；无新重复替换整句、无新增Markdown标题损伤，改写样本保留监督token。全套186项测试通过。前向/反向诊断仅验证有限loss和梯度，不做optimizer更新，不生成checkpoint，诊断loss不是第五次训练成绩。验证记录`outputs/target-revision-v3-review-20261005/setup-validation-final.json`。

训练运行后，新目录的experiment.json会记录实际v3.1清单与train哈希；loss逐步写入losses.jsonl，逐轮写metrics.jsonl，状态写status.json。完成后再加载第五次best.pt，保持相同135条清单和生成设置，按风险v1.1/其他v1.0评分，分开原27、新108、总135，重点报告退款相关指标及其他intent副作用。不得重新挑dev问题或读取sealed_test。
