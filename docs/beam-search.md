# 第五模型：束搜索学习演示（不训练，不接检查器）

本轮问题只是理解“同时保留多条续写路径”。仍用第五best.pt、冻结tokenizer和原角色屏蔽/context512/max_new_tokens320，不改旧greedy实现、评分或训练结果。演示固定已有135清单中的cancel_order/change_order各第一条，选择规则在生成前确定；两条不构成135条质量对照。

## 原理及本实现的契约

greedy每步只留1条；beam_width=4每步至多留4条。每条未结束路径各扩展其4个最可能的合法token，再将所有扩展与已结束路径合并，统一排序留4条。完成路径占束位但不再送入模型。宽度1与greedy一致；宽度4不是四次随机采样，也不保证四种不同意思。

举例：第一步A概率0.55、B概率0.45，greedy选A。若后续EOS概率A为0.40、B为0.99，两条完整路径概率分别0.22和0.4455；束搜索可以保留B并在后续排到前面。这是**模型序列概率更高**，不是B必然答对客户诉求。

累计分数使用logP=sum(log p(token|问题+已有回复))；排序分数=logP/(生成长度**alpha)，默认alpha=0.7，长度含EOS不含问题。正alpha减弱负logP累加对长回复的不利影响，不是质量奖励。角色token禁止选择，概率分数使用屏蔽前原模型softmax，不重新归一化。平分按token ID序列字典序固定。不同alpha的score不能直接互比。

这是便于教学的**固定束池＋前缀长度归一化**版本：中间裁剪也使用上述分数，已结束路径占槽，没有独立finished池、标准库early-stopping/KV cache或全局最优保证，不声称逐位等同Hugging Face beam search。early EOS会使可探索路径减少，剪掉的路径无法恢复；增大beam未必改善任务或多样性。

EOS/上下文上限/token上限分别记录，不人为补EOS；EOS优先。每步trace保存parent_rank、扩展token、token logP、累计logP、排序分数、长度、是否携带已结束路径，可以沿父索引重建各条路径。候选最终保存原文、完整token、分数、停止原因及不同文本数。

## VS Code 运行

打开“项目工作台.code-workspace”，在“终端→运行任务”选“第五模型束搜索演示（2条问题，保留4候选，不训练）”。普通仅打开文件夹时看不到这个工作台级任务。旧.vscode/tasks.json已绑定训练冻结哈希，本轮不改它；不要用Ctrl+Shift+B（仍是训练入口）。演示结果会显示在终端，每次只写新outputs/beam-search-fifth-demo/时间-UUID目录。

也可在项目根目录、已选项目Python的终端运行：python -m my_project_1.beam_search --config configs/beam-search-fifth.json。配置beam_width、num_return_sequences、length_penalty解释见上文；本轮不是参数扫描。

## 把两种收益分开

每条机器记录有两个独立区域：

1. candidate_quality：greedy质量、beam第一候选质量、候选池是否含可接受答案。当前status=not_scored、各判断为null；运行成功、分数较高不等于质量改善。以后先固定病例/判据，再按同一标准评greedy、top1及整个候选池。
2. checker_selection：检查器选哪个、是否正确、相对beam top1收益。当前status=not_implemented、各值null；不将没做筛选记0分或成功，不因人工读到好候选就说检查器有效。

之后如获授权实现诉求检查器，应冻结同一候选池，区分“候选池没有好答案”和“有好答案却筛错”，并记录全不合格时拒绝/澄清。检查器筛选收益不能算成beam或模型学习能力改善。

## 2026-10-07真实功能演示

35项束搜索/greedy单测通过；增加真实保存trace逐步重建、观察引文及分阶段空值、旧冻结和工作台入口检查后，共37项通过（3.18秒），后两项只读不新增推理。第五真实模型两问题各返回4个EOS候选，权重及梯度检查通过，greedy逐字复现第五历史两条。实际目录outputs/beam-search-fifth-demo/20261007T021508-39fb54baab4a4a1e9231dbe8fbe0f59b，candidates.jsonl包含完整回复/trace，summary.json记录执行；demo-observations.json单独记录两病例动作范围观察，completion-freeze.json绑定证据。

取消整单病例的greedy和beam都响应取消，没有由此证明质量提高。删除部分商品病例中，greedy追问要删哪些商品，4个beam候选却都给取消整单流程；候选虽文本不同，却没有范围正确的答案可供检查器选择。**该病例反而展示概率排序可能损害任务匹配。** 这是已知模型的两条全文AI范围观察，不是五维评分或总体胜率；没有据此改默认greedy、扩大束宽或接检查器。生成候选的观察与未实现的筛选效果保持分开。
