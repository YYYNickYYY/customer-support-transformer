# 第五次副作用审查：结论与下一步

2026-10-06。完成27条生成案例（24条重复0→1，3条残余退款保证）、48条固定训练抽样完整v2/v3.1文本、7组残余保证上下文全文审阅。协议和选择在展示目标前冻结；结果在`outputs/fifth-side-effect-audit-20261006/`，原模型、数据、评分及工作簿没有修改。

## 结论

1. **训练目标仍有同类结果保证，属于已确认的覆盖遗漏。** 不是所有残余都能解释为模型凭空重组。
2. **多数显眼重复组合没有精确训练命中，且24条来自第五次没有改动目标的12个intent。** 不应立刻把它们称为第五次改写接缝，更不能批量删除正常服务意愿。共享参数变化仍可能影响其他intent，本诊断不证明独立因果。
3. **“新增重复24条”至少有一条评分尺度争议。** 当前数字是历史已发布快照，不能将全部24条当成已经确认的新缺陷，也不静默覆盖原分。
4. 目前不启动第六次。先统一重复评分，再决定下一训练因素；若后续继续数据实验，优先完善**同一退款结果保证因素的上下文提名及混合句覆盖**，不与去冗余或超参数同时修改。

## 固定样本与验证

- 保留全部24条重复变化及3条保证病例，两模型全文、问题、分组、证据与停止原因。
- 训练抽样按SHA256(`fifth-side-effects-v1:42:`+ID)排序，各相关intent全体取3；申请/进度退款另取已改和未改各3，去重留层标签，合计48唯一ID，不按文本好坏抽样。
- 核验v3.1所有manifest/files/dependencies、评分源哈希，以及21528条v2/v3.1完整对齐、非response字段一致、修改ID与changes一致。规则成功回放不代表语义风险全部清零。
- 所有抽样及上下文完整阅读后保存逐条明确结论；新增抽样稳定性/文本无关性测试与评分相关测试共9项通过。不是全训练语义普查，也不是独立人工评审。

## 数据遗漏：具体证据

以下ID省略共同前缀`bitext:fb86d1b60038970d79fefbfa0c28b47c22d8b961:`。

| 训练ID | 原文片段 | 发现 |
| --- | --- | --- |
| 16362 / get_refund | `assist ... ensuring you receive the refund you're entitled to` | 未改目标仍保证获得退款。协助分句不能抵消同句保证分句 |
| 16468 / get_refund | `ensuring that you receive the appropriate compensation` | 未改目标仍保证补偿获得；compensation变体不能因没写refund就排除 |
| 16829 / get_refund | `let's work together to ensure a smooth resolution` | 退款上下文确定解决结果，缺努力程度限定；保守执行谓词筛选未覆盖 |
| 26729 / track_refund | `expedite the process and ensure a smooth resolution` | 与第五生成26320/26438类似的结果保证在未改目标中存在 |
| 26728 / track_refund | `ensure a smooth resolution for you` | 退款进度上下文保证解决，未改 |
| 26327 / track_refund | `Your cooperation ... will ensure a smooth resolution` | 明确肯定业务解决结果，未改 |
| 26516 / track_refund | `ensure that everything is in order with your rebate` | 已改响应末尾到账保证正确替换，但前句业务状态保证未处理；原有残留，不是改写新增 |
| 26033 / track_refund | `actively working on processing your compensation` | 未改目标宣称正在处理补偿，且模拟查状态；处理断言与一般努力分开审阅 |

需要保留/澄清的反例：26822为`we'll do everything we can to ensure ...`，属于努力限定，不能一刀切删除；26056的`help us ensure a smooth resolution process`接近流程体验目标，暂列边界复核而非自动批准修改。7组上下文中5组按当前结果边界列明确保证、1组边界、1组正常努力；只是此词组检索的7组，不是全train风险总数。

六条抽到的已改目标中，批准片段均正确替换，没有看到新引入重复核查句或语法接缝；必要订单/交易号、条件与正常work towards保留。26516的未处理前句说明“批准片段全部命中”不等于“同响应其他保证全部覆盖”。

## 为什么旧筛选漏掉

核查实际冻结提名及筛选程序，没有改动它们：

- `propose_target_v3.py`依赖句内处理/收到等谓词与refund/reimbursement/rebate共现，以及rectify等词；退款响应中的纯`ensure ... resolution`句、compensation变体可能没有进入提名。
- `review_target_v3.py`先判断QUAL、再PASSIVE/EXEC；只要出现`assist ... in`等宽泛协助标志就可能整句排除。16362的完整句以`qualified-service-or-possibility-retained`排除，后续没有显式AI审阅记录。该句同时明确保证退款获得，属于过宽排除。
- 16829的保证句以`no-explicit-refund-execution-predicate-conservative-exclusion`排除。已冻结方案原本公开不保证所有语义风险清零，本次找到了具体缺口，而不是训练没有使用最终清单。

修正原则：提名时结合退款上下文及补偿同义词，宽召回；混合句拆清正常协助与确定结果，阅读全文裁决。不能把所有ensure/help或所有compensation句自动改掉，也不能只扩大正则后直接训练。

## 重复归因：数据与生成都可能存在

第五新增重复24条的12个intent均不属于第五修改的四类（退款政策、申请、进度、取消订单）。这些intent的目标文本v2→v3.1没有被改写，不能归为它们的新增文本接缝。

完整片段`form or a form`、`smooth transition and ensure smooth transition`、`email ... newsletter email ... newsletter`、`Double-check ... for accuracy`等在全v3.1精确检索0命中，支持“非逐字照抄该组合”的有限线索；不证明没有语义来源或纯模型原因。单独感谢语476命中、体验话术28命中，不能证明模型重复排列是训练直接教的。

固定样本也有原生问题：22821的反馈到达部门目的重复，25707的`{{Order Number}} {{Order Number}}`原生实体重复；4238/3993等有实体替换；但许多目标是正常步骤、必要澄清或不同作用的礼貌句。没有证据支持全局删除服务话术能解决全部生成循环。

原27/新增108与每intent的病例/层标签均在机器记录保留。抽样规模不支持估计全train冗余率；不能由全train短片段命中数量推算风险率。

## 重复评分尺度争议：必须先解决

8916 / contact_human_agent：第四及第五都包含“To connect...请给具体问题...better understand issue...”的同一核心复述，却分别重复0和1。文字微变不足以证明该冗余首次出现。7011、22756也接近前后认真处理/分享想法的敏感边界，列复核候选，不直接改分。

目前只标争议，历史24条及55→66快照保持原样，不据此估算“真实新增23条”。要修正尺度，应另存一个统一复核快照：**第四、第五同135ID的全部重复字段，模型标签隐藏混排，沿用v1.0**；保存原分/新分/原文证据/理由，汇总原27、新108、总135及配对。其他四项不顺带重评。这是AI复核，仍不是严格独立盲评。

## 后续任务顺序（尚未执行）

1. 统一复核上述第四/第五全部135条重复字段，排除尺度漂移后判断重复副作用幅度。
2. 为同一退款结果保证因素制定上下文提名/混合句覆盖草案，审阅候选全文、保留努力限定及流程体验反例，冻结完整修改清单。不要顺带添加查询能力、实体纠错或订单澄清改写。
3. 若用户决定第六次训练，只改变批准后的同类覆盖与整句完整性，保持5轮、学习率、模型和生成配置；由VS Code启动。去冗余、解码诊断、上下文长度留作另外的实验，不同时混改。

本轮没有训练、修改目标、更新工作簿或评分、读取sealed_test或推送GitHub。
