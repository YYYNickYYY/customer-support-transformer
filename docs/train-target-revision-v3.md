# 第五次唯一因素：退款执行／补救结果承诺

基线为冻结v2。仅改train.response中未经后台核实的退款执行、退款完成/到账及退款语境的确定补救结果，采用整句改写；不改instruction、ID、实体、样本顺序、dev、tokenizer或模型。

全train扫描候选，不按dev成绩筛选。提名不等于批准：审阅候选唯一整句，必要时核查原响应上下文；词法筛除与AI审阅分别记录，不宣称全量训练文本已无语义风险。正常努力意愿（try/aim/work towards/do everything we can）、可能性、退款资格/政策条件、联系支持与自助导航保留。查询准确性/后台查单、隐私、订阅、泛化满意保证不属本因素。混合句必须保留请求材料、条件及实体，不能整句机械删掉。

每个提名句记录全部成员ID、原响应、替换整句或拒绝理由；批准后精确回放，冻结proposals/approval/rules/changes/train及依赖哈希，拒绝覆盖。实体占位符多重集不变，不新增模板组或样本，保留已有v2干预。

5轮、batch8、lr0.0003、weight_decay0.01、grad_clip1、seed42、cpu_threads2、设备auto、模型及生成配置保持第四次相同。第五次从相同seed重新初始化，不从第四次继续训练。冻结完成后才允许VS Code任务启动；不得读取sealed_test。
