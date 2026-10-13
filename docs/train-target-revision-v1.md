# 第三次训练目标干预协议 v1

协议：`result-privacy-target-v1`，2026-10-05。先冻结本协议和精确规则，再冻结修改清单，最后训练。

## 唯一因素

改变训练回复中的确定处理结果/隐私处理保证，改为核查或协助表述。不改instruction、ID、顺序、样本数、intent/category及其他字段，不删除样本，不改冻结train/dev/tokenizer。源train SHA-256须与冻结tokenizer绑定值一致。sealed_test不读取。

不按ensure关键词自动判风险，不按27条dev题目或模型表现选样。规则对全train所有intent一致应用。此次是保守的明确片段干预，不声称覆盖训练集所有保证话术。

## 确认规则

1. `ensure/ensuring a smooth / satisfactory / smooth and satisfactory / fair / swift resolution` 分别改为 `work/working towards a resolution`。确认依据：ensure直接指向解决结果，超出单纯体验目标。
2. 第一人称复数 `we will/we'll handle your [personal] information with the utmost care [and protection]` 改为 `we recommend sharing only the necessary information through an official support channel`；并列谓语 `and will handle...` 对应 `and recommend...`，保留句子原主语。只接受片段后为句末/标点，避免残留and protection等造成拼接语病。这是安全渠道建议，不保证已安全处理。
3. `ensure that your subscription is promptly canceled / canceled promptly`，及 `ensure your subscription is canceled promptly` 改为 `help you submit your subscription cancellation request for review`。若原句随后明确接 `, and you will no longer receive our newsletter`，或 `and that you won't receive any further newsletters`，将该结果保证一起纳入精确长片段，优先于短片段替换。这是协助申请而非取消成功保证。

所有规则按精确文本、完整词界匹配，忽略大小写但保留句首大写。隐私规则较长变体优先，禁止重叠替换。

为保留正常服务意愿，命中句同时出现goal/aim/strive/try、do ... best、make ... effort、work ... towards、work diligently/together/with you、do everything we/I can、do my/our/your utmost、help ensure、want to ensure、may/might时，本次保守不改。`your cooperation/insight/input/feedback/assistance`为客户配合促进处理的目的表达，也保守排除，避免把名词主语改成会“work”的主体。确定性退款、配送事实、PIN/账户操作流程、一般体验/满意目标、实体、语病、重复和长度均不纳入本次干预。`will be handled with utmost care`可能描述投诉而非隐私，不纳入隐私规则。

这些有限片段在源train完整句上下文中核查后予以规则级确认，实际命中全部处理；并非每行独立人工风险代评，也不将词法命中比例称作风险率。记录未命中和被服务意愿排除的候选，不为提高覆盖率扩大规则。

## 冻结与核验

衍生目录 `data/derived/result-privacy-v1/`：完整train.jsonl、changes.jsonl（每个修改ID、原/新回复、逐片段位置/原文/改文/规则/理由）、controls.json（固定seed42规则内最多5条确认样本及不改反例）、manifest.json。

manifest保存源/衍生train、修改清单、控制样本、协议、规则、实现代码与dev/tokenizer/model/生成配置的哈希，最后写入frozen状态。先验证完整行数、ID和所有非response字段，再写冻结标记。目录存在就拒绝覆盖。

训练加载须重新校验原train、manifest相关哈希与精确规则重放，逐条核对衍生response和修改清单，冻结不完整或篡改则拒绝训练。不修改tokenizer绑定的原split哈希；训练experiment.json单独记录衍生源和实际train哈希。

## 固定参数与观察

5轮、seed42、lr0.0003、batch8、weight_decay0.01、grad_clip1、cpu_threads2；模型4层/d_model128/4头/FFN512/dropout0.1、词表4000、上下文512均固定。从同一随机初始化训练，不从第二次best.pt续训。dev保持原始2688条，选择最低dev loss的best.pt。

新目标长度可能自然变化，是文本干预的后果，不额外调整上下文或样本筛选。检查保留EOS/监督token和衍生截断变化。训练后同27条生成配置不变（greedy、320新token），按风险v1.1及其他v1.0维度重新评分。

原dev参考中存在保证话术，谨慎回答可能得不到更低loss，需结合风险与帮助评分。协议受dev诊断启发，不能把此次结果当独立泛化证据。
