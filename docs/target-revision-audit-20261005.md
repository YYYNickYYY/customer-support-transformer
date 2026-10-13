# 第三次训练目标改写诊断

日期：2026-10-05。只读诊断，未修改冻结数据、模型或训练参数，未启动第四次训练。下一次正式训练由用户在VS Code中启动，不在Codex后台启动。

## 结论

**已确认有数据层面的覆盖不足及局部替换接缝问题，同时存在原生文本缺陷和模型生成重组线索。不能单独归因于学习率、训练轮数或模型容量。下一步优先建立同类保证话术的完整句改写v2，先冻结清单再训练。**

原v1是有限、保守的精确片段干预，未宣称处理全部风险。这里的遗漏指同类保证仍留在衍生目标中，不是文件损坏或冻结规则未执行。不得覆盖v1，也不将本审查当作新一轮模型评分。

## 方法、范围与证据保存

- 预先保存[诊断协议](target-revision-audit-protocol-v1.md)，再冻结抽样ID：开户/newsletter两类，分别在全体、已改、未改三层按SHA256(`target-revision-audit-v1:42:`+ID)取前5条。30个层内位置去重后20个ID，反例未替换。
- manifest所有文件与依赖哈希均一致，全部21,528条精确规则重放与衍生数据/修改清单一致。260条、260片段实际执行，无非response字段变更。
- AI读完全部260个改写句的before/after及前后句，保存逐ID审查；不是仅看几个好例子。完整before/after也另存，未称260条全文业务风险都已独立人工评分。
- 全衍生train做预定义宽松句级筛查，保存目标两类及已改行的727个候选上下文。全部候选句已阅读，边界/代表候选的邻句及固定20条全文进一步复核。没有把727候选全部当作完整回复逐条业务代评，也没有计算“全train真实风险率”。词法共现可能是服务意愿、密码建议或其他任务。
- 判定参考冻结风险v1.1边界，评审者为AI、已知第三次结果；无商户安全政策、真实流程或工具记录。保证缺依据不等于已证明虚假。
- 未访问sealed_test。26项诊断工具与冻结改写测试通过，旧冻结文件/标准/规则不变。

产物目录：[诊断记录](../outputs/target-revision-audit-20261005-v1/)。其中selection.json保存预先冻结ID/哈希；changed-contexts.json和complete-changed-responses.json保存260条上下文与全文；selected-samples.json保存20条全文；candidate-contexts.json保存候选；ai-context-review.json保存逐条理由、证据、反例和检索结果。分析程序为`scripts/audit_target_revision.py`与`scripts/finalize_target_audit.py`。

## 1. 开户和退订的未覆盖变体

| 已核查的同类变体 | 示例ID末尾 | 衍生目标中仍有的原文 | 原v1未覆盖原因 |
| --- | --- | --- | --- |
| 被动处理 + data | 9915 | Your data will be handled with the utmost care. | 只改固定第一人称主动句 |
| 第三人称信息 | 10242 | your wife's information will be handled with the utmost care. | 对象不是固定your information |
| 代词宾语 | 10343 | will handle it with the utmost care. | 未解析it回指information |
| 额外并列尾句 | 10177 | we'll handle your information with the utmost care and confidentiality. | 原规则要求句末/标点，并列变体未纳入 |
| 省略the | 10391 | we'll handle your information with utmost care. | 精确词串不同 |
| 换guarantee动词 | 10694 | guarantee that the information shared will be handled securely. | 不在固定handle句型中 |
| personally、英式双l | 17011 | I will personally ensure that your subscription is cancelled promptly. | 插入副词及cancelled词形 |
| 去掉that、调整副词位置 | 17046 | I will ensure your subscription is promptly canceled. | 不等于原规则中几个固定词序 |
| assure/trust尾部并列 | 17853 | You can trust that we will handle your information with the utmost care and respect your privacy. | 片段尾部并列未纳入 |

上述编号是原始ID末段；完整ID和原问题/上下文均保留在JSON，不另创训练ID。

两个可复现的精确子串统计（大小写不敏感、不是语义风险计数）：

- `information will be handled with the utmost care`：原train62条，衍生仍62条，其中开户53、newsletter9。不能把62与其他候选相加当独立风险样本。
- `handle your information with utmost care`：原train14条，衍生仍14条，其中开户4、newsletter10。

固定未改层反例保留：开户5条中3条明确隐私处理保证，另2条未见同类保证；newsletter未改层5条中2条有明确取消处理保证。样本很小、分层有重复，不能外推发生率。`prioritize protecting`、`Our aim ... seamless experience`、`do everything I can`、`smooth cancellation experience`不因关键词命中就改成风险。处理request的care也不自动视作隐私信息处理保证。

## 2. 260条中已经改过的行仍可能留有保证

- ID **10148**：隐私句已换成渠道建议，但前句仍为 `Once I have these details, I'll ensure that your account is set up smoothly and securely.`。安全开户结果仍被确定保证。
- ID **17513**：隐私句已改，前句仍为 `This will ensure that we accurately identify your subscription and promptly unsubscribe you from our mailing list.`。不是单纯“协助退订”，仍确保准确识别及成功退订。该例也命中预先固定的已改层样本，非只按输出挑选。
- ID **17654**：取消结果改写后，`Please note that this action is irreversible...` 仍保留。这是原有业务政策断言，不是同类结果/隐私话术v2需要顺带清洗的对象；对应changed-context索引204。

只把命中片段替换正确，并不能证明整条回复已没有同类保证。v2应检查同句及相邻句，而不是一个response遇到第一个匹配就结束。

## 3. 替换新引入的接缝问题

| ID末尾 | before → after | 诊断 |
| --- | --- | --- |
| 26329 | while we work to ensure a satisfactory resolution → while we work to work towards a resolution | 明确新增重复work to work，源文本没有这段重复 |
| 15952 | This will expedite the process and ensure a fair resolution. → This will expedite the process and work towards a resolution. | This指做法/配合行为，改后让它自己work，主体与动作不匹配；应明确we动作 |
| 7833 | To ensure a smooth resolution process → To work towards a resolution process | 目标从顺利解决流程变成“争取一个流程”，搭配不自然 |
| 18809 | 同上一种resolution process变体 | 相同目标搭配问题 |
| 18775 | working closely with you to ensure... → working closely with you to work towards... | 轻微句式冗长，不是残句；collaboration与goal仍有不同信息，不按work词频自动当严重错误 |

记录了5个需整句复核的措辞点，其中包括1个较弱风格问题；**不能称为5条确定语法错误**。其余255个改写上下文未观察到明确新接缝问题，不代表它们全文均无语病或无风险。

同时发现排除式匹配对`work closely`、`doing everything we can`等努力表达不够完整。v2需重新判断这类上下文是否只属服务意愿，不能为了更多命中扩大保证定义。

## 4. 数据与生成不能混为一谈

| 第三次现象 | 精确检索 | 可支持的解释 | 不能推出 |
| --- | --- | --- | --- |
| 开户生成 `we'll handle your information with the utmost care.` | 原train5、衍生0；其他主语/被动/data等变体仍大量存在 | 目标仍教授相似保证，模型可能泛化组合出已移除的词串 | 不能证明记住某一条原样本；也不能说完全是模型独有问题 |
| 退订生成 `your privacy and will be handled with the utmost care.` | 原/衍生完整子串均0 | privacy态度句与被动处理保证的生成重组线索 | 不命中不证明没有近似训练表达，更不证明因果 |
| `any of any` | 原/衍生均0 | 模型生成拼接线索 | 不能据此把所有重复归因模型 |
| `account!I` | 原/衍生均433，均edit_account | 原生训练语病已有来源且本次未清理 | 不能证明该模型输出直接记忆其中一行 |
| 地址录入中混合两种报错追问的完整句 | 原/衍生精确句均0 | 生成重组/任务偏移线索 | 不能以精确不命中代替语义来源分析 |
| 退款政策末尾未完成 | 已记录320新token上限 | 生成上限截断，与本次改写接缝分开 | 不直接等同模型语言能力不足 |

账户权益、真实表单、自助注销、PIN流程、实体替换、原生重复等保持独立归因；此次v2不一并修，否则不再是单因素实验。

## 下一步方案与VS Code运行

**分支选择：已经确认数据问题，优先v2，不需要先跳到“目标完全正确、仅扩大dev检查”的分支。** 更大固定dev评估仍有价值，但不能代替修复已确认的目标缺陷。

建议v2：以第三次衍生train为基线，仅完善**同类处理结果/隐私保证话术的覆盖与完整句改写**。以完整句/并列分句为单位处理，明确动作主语，并检查邻句残留；保留姓名/订单/信息对象和正常服务意愿。不能无限制删ensure或把所有“secure password”当隐私保证。

v2清单需区分新增改写、对原v1措辞的修正以及保守排除。订阅成功/邮件投递与取消成功不是同一已冻结具体规则，是否扩展到前者必须在新协议中显式决定，不能暗中改变范围。此处先建议维持原“解决结果、取消处理结果、隐私处理保证”三类；安全开户表述按隐私保证语义完整性纳入上下文审查。

保持5轮、lr0.0003、batch8、seed42、模型/tokenizer/上下文及greedy/320设置不变，从相同初始化训练。先冻结v2精确规则、逐IDbefore/after、反例和哈希，再添加第四次VS Code任务。**本轮仅完成诊断，尚未建立或冻结v2，也未创建指向不存在v2数据的训练任务。** 不应点击第三次任务重复训练旧清单。

第四次任务就绪后：VS Code中按Ctrl+Shift+P → Tasks: Run Task → 第四次训练任务；实时输出在“终端”，不是“调试控制台”。不采用后台训练加日志查看的旧方式。此约束已记录到实验日志，下一步实施须遵守。
