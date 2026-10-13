# 第五次模型独立生成诊断

2026-10-06。结论：已用第五次实际best.pt复现并追踪14条固定dev病例、2273个生成token。所有正文、token ID、token数与停止原因精确匹配旧输出。保证和重复病例均正常EOS结束，不能归因于320截断；另4条政策回复确实被320上限截断，但不是512上下文不足。**不训练、不扩大改写、不调整参数或解码设置。**

## 1. 本次做了什么

先冻结[诊断协议](fifth-generation-diagnostic-protocol-v1.md)，再固定病例和源码/模型/原文哈希。3条已知退款保证病例、哈希抽出的2条新增重复和2条持续重复、3条风险0/重复0参照，另加全部4条非EOS输出，去重后14条。病例不是随机总体效果评估，参照也不是同intent匹配因果对照。

模型：`runs/20261005T132346-full-ebf624ab40734b858b46553b4da37537/best.pt`，SHA-256 `2a2e549f954d96325daf4cb5c286edae9f9441b2ea806bab0bcdd1734fa46479`。这是实际训练v3.1的第五次模型，从未使用v1.2预览训练。

沿用CPU、2线程、seed42、冻结4000词tokenizer/模板、上下文512、greedy、320新token、屏蔽PAD/BOS/USER/ASSISTANT；EOS不屏蔽。无参考回复输入、梯度或optimizer。记录每步top5、选中概率、EOS概率/排名、top1与top2的logit差，并把原文证据定位到生成token位置。

最终诊断产物在`outputs/fifth-generation-diagnostic-20261006-v1.1/`：selection/freeze、14份traces、summary、interpretation-events、analysis及analysis-freeze。

版本修正：v1在比对旧记录prompt_tokens时发现旧评估未保存该字段，停止并保留冻结材料；v1.1新增兼容入口，仅在内存按冻结tokenizer补算prompt长度。14病例清单不变，旧记录与旧脚本未修改。新正文/token/停止原因逐项比对成功；prompt长度是本次导出，不冒称旧文件已有记录。

## 2. 保证话术是如何生成的

下列token位置从生成回复第1个token开始，包含最后的EOS。概率是屏蔽四个角色控制token后，对当前前缀的下一token分布，不是业务正确性置信度。

| 病例 | 观察到的生成片段 | token位置 | ensure开始时概率 | 最终结束 |
| --- | --- | --- | ---: | --- |
| 16355 | ensure a smooth and satisfactory resolution | 27–32 | 0.5739 | EOS，第71步 |
| 16355 | ensure a smooth and efficient resolution | 64–69 | 0.2152 | 同上 |
| 26320 | ensure a smooth resolution | 91–94 | 0.4165 | EOS，第104步 |
| 26438 | ensure a smooth resolution | 75–78 | 0.4346 | EOS，第88步 |

16355在普通“guide you through the process”之后，第26步and获胜，第27步ensure获胜；后续又在guidance and support之后回到ensure结果保证。26320/26438都从“Your insights will help us expedite the process”继续and ensure，再接smooth resolution。

26320在resolution分支，模型分配resolution约0.3880、experience约0.3234；26438约0.3913与0.3619。这两个单词在v1.1评价中语义边界不同，但模型只按下一token分布选择前者。本次没有改成第二候选生成，不声称换一个词就能解决整体风险。

这些保证发生在结束前几十步，不是截断补全造成。普通协助前缀与结果保证在同一回复里共存，说明“出现help”不是风险豁免；模型没有通过这份训练获得真实查询权限或业务事实。

第五实际train的区分大小写精确子串计数：完整`ensure a smooth and satisfactory resolution`为0，完整`Your insights ... ensure a smooth resolution.`为0，`ensure a smooth and efficient resolution`为8。0只表明非该完整字符串逐字复现；不能证明没有训练语义来源，也不能唯一归因到某一条样本、注意力头或第四→第五因素。

## 3. 重复不是同一种问题

| 病例 | 类型 | 形成位置与观察 |
| --- | --- | --- |
| 24647 | 同句逐字重复 | 第164–167步首次ensure a smooth transition；168步and；169–172步原样再说一次 |
| 17634 | 局部词语回环与语病 | 退订替代路径已说reply to the email，第63–72步又生成with the email to the newsletter email with the newsletter |
| 21189 | 相邻句语义重复 | 第4–5步proactive approach作感谢；第17–18步又在下一句感谢同一积极反馈 |
| 7062 | 间隔语义重复 | 第27–30步take your concerns seriously；必要澄清后，第72–75步又take customer claims seriously |

24647第二次ensure的概率0.2927，第二候选make为0.2074；之后smooth与seamless分支也接近。17634在第65步email约0.2376，次选词表片段wor约0.2179；66步to约0.3954、with约0.3829。这里能观察到局部greedy分支竞争及反馈后的后续走向，不能凭小间隔证明“随机采样一定更好”。

21189/7062则表明重复不总是低概率或逐字循环：第二次approach/seriously的选中概率约0.9924/0.9898，模型仍可高概率续写语义冗余。四条均最后正常EOS结束，不是无限生成卡死。

训练逐字检索：24647完整双句重复0，但单次ensure a smooth transition有148条；17634完整回环0，但reply to the email有2条；proactive approach有297条，21189第二完整感谢句0；take your concerns seriously有36条、take customer claims seriously有7条，同时含两者的同一train记录0。

这支持“常见话术片段可能被重组”的线索，**不是内部机制或纯模型因果证明**。本次未做attention归因、反事实前缀实验或训练影响估计。

### 为什么不能直接禁止所有4-token重复

字面4-token检测没有标出17634的email回环，也没有标出7062的语义重复；11494正常重述删除账户对象的`{{Account Category}}`却出现3个重叠4-gram重现。这是同一必要实体的不同功能复现，不应当作语义重复扣分。

因此ngram检测只作排查线索，不重新打分；不能一刀切去重占位符，或仅因某重复病例消失就说解码整体改善。

## 4. EOS与长度上限

全部同135旧输出：131条EOS、4条max_new_tokens。4条均是check_refund_policy长回复。本次全部纳入，而非只挑最坏或最好的截断病例。

| 病例 | 320步时仍可用上下文 | 最后token | 当步EOS概率 | EOS排名 |
| --- | ---: | --- | ---: | ---: |
| 6483 | 181 | anything | 5.24e-8 | 180 |
| 6382 | 176 | you | 1.58e-5 | 29 |
| 5994 | 181 | than | 9.97e-9 | 361 |
| 6262 | 184 | Is | 8.37e-6 | 99 |

所以直接停止条件是320，而不是512上下文用尽，也不是EOS被错误屏蔽。5994切在“I'll be more than”内部；其他3条已经完成协助结语，却继续生成额外“Is there anything…”并截断。

例如6382完整assist you further.后第311步换行概率约0.9081，EOS约0.0413（第2名）；6483对应第316步换行约0.8072，EOS约0.1189（第2名）；6262第318步换行约0.8737，EOS约0.0443（第3名）。greedy正确执行了继续写的分布选择，本次不把生成器实现误认成停止规则故障。

这不证明加大320一定值得：政策类回复本来就较长，部分询问页面入口的问题却展开六类政策条款，还加礼貌尾句。增加长度可能仅补完尾句，不能修复帮助、意图或虚构政策。本轮没有延长或提早终止，未验证任何新长度效果。

## 5. 总体解释与下一步

确定观察：复现14/14，2273个token；保证/重复发生于正常EOS前；4个截断病例上下文尚有余量；存在逐字和非逐字多种重复；模型的续写概率不等于正确、帮助或安全。

合理线索：共同模板片段与局部续写反馈可能参与话术重组，greedy容易继续模板尾句。尚未证明：具体训练样本造成某个错误、数据改写导致重复变多、模型大小/学习率是根因、新v1.2可修复旧输出。

控制样本也不能当全面优秀：866风险0/重复0，但客户不能付款，模型回答取消订单，原评分意图0。本次不改分。这说明减少风险和重复仍不足以证明总体能力提升。

下一步建议保持第五checkpoint，另行预登记一次**仅解码重复约束的受控诊断对照**，不重训、不同时增长度。须在同135上同时检查语义重复、意图、帮助、通顺、风险、实体保留和EOS，明确4-token约束会漏掉哪些问题、误伤哪些正常复现；不能只对24647成功就全面启用。该对照本轮尚未授权或实现。

长度、EOS偏好和事实约束分别是不同因素；必要时各设独立实验。若接下来仍选择数据干预，也必须重新取得训练授权并使用已审核清单，不能把本次诊断当第六次启动许可。

## 6. 验证与保留

生成器与observer契约测试检查角色token屏蔽、EOS/320/上下文优先级、反馈、非有限logits、模式还原与byte-BPE证据定位；实际trace检查对应原文、长度、哈希及零optimizer更新。完整项目测试本轮229 passed（100.19秒），已在单一实验日志追加。

原checkpoint、训练数据/配置、tokenizer、原135生成与评分、v1.2清单及所有旧表保持不变。没有sealed_test读取、GitHub推送、新checkpoint或训练。首轮单元测试torch导入时曾打印Windows WMI异常栈，但进程随后完成25项测试且exit0；后续实际推理正常，不修改环境来隐藏该提示。
