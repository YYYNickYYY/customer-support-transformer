# 第六次学习率单因素实验 v1（冻结，尚未启动）

日期：2026-10-06。问题：在相同小模型、相同第五次监督数据与训练预算下，将 AdamW 学习率从 0.0003 降至 0.0001，是否改善整体 dev 表现和回复质量？不追求逐条修复，也不从既有局部诊断推断学习率一定是原因。

## 唯一变化与运行边界

只改变 learning_rate，仍训练 5 轮、batch_size=8、seed=42、CPU 两线程；weight_decay=0.01、grad_clip=1.0。模型仍为 4 层、128 隐藏维度、4 头、FFN512、dropout0.1、词表4000、context512。冻结 tokenizer、角色模板、回复/EOS监督与动态 padding 均不变。

使用第五次实际训练数据 refund-execution-v3.1/train.jsonl（21528条），dev2688条。未启用退款补充清单，不改数据，不读取 sealed_test。用同一个原训练引擎重新设置 Python/PyTorch/打乱种子并构造模型，**不加载第五次 best.pt，不续训**。相同种子/代码/CPU环境确保起点与打乱契约相同；不宣称跨硬件逐位可复现。

独立配置 configs/training-sixth.json；启动入口 my_project_1.training_sixth。必须显式选择 --check 或 --train，无默认训练。启动前校验下方全部哈希、实际数据完整重放、第五次记录、设备及 PyTorch 版本；变化或核验失败就退出，不自动修改配置或重冻结。--check 不构造模型、不加载权重、不执行 forward/optimizer、不创建 runs 子目录。协议本身不能自哈希；核验报告输出其 SHA256，启动时保留终端记录。

## 固定比较，不扩大诊断

与第五次同一135条 dev（27 intent×5，顺序/ID/问题固定）比较。第五次 best.pt 为第5轮，最低 dev loss=1.1029298457139332；第六次仅按完整轮次最低 token 加权 dev loss选 best.pt，不按135条得分挑 checkpoint。

训练完成后另行授权加载新 best.pt，保持下方生成配置，**仅替换 checkpoint 路径及其哈希**：CPU两线程、seed42、greedy、屏蔽PAD/BOS/USER/ASSISTANT、保留EOS、max_new_tokens320、不加重复约束或人为前缀。135条完整生成一次；不换病例/候选/解码。

主要看最低 dev loss、意图与帮助；同时报告风险条数、重复条数及通顺均分。风险用v1.1，其他用v1.0。第五次基线使用重复复核后的权威JSON：重复63条、风险43条，不沿用旧重复66，也不混入其他解码校准的风险42。具体均分见下方机器清单。

评分尽量混排、遮蔽模型名并提供完整原文；逐条配对给证据。若有边界校准，两模型一起复核，另记原因，不静默覆盖冻结历史。评分属于AI辅助诊断，并非独立人工盲测。dev已用于选择和多次实验，结果仅支持本项目这组固定样本的实用比较，不是通用客服能力或统计显著性。

## 决策与停止规则

先列第5/6次的主指标及副作用，不折算一个总分，不因个别漂亮回复采用。仅当意图、帮助的总体变化支持收益且风险/重复/通顺没有明显退步时，结合dev loss判断是否值得采用；loss更低但任务表现没有改善不算成功。报告不确定性和逐条改善/退步数量；不预设必须击败第五次。无清楚整体收益或存在重要副作用，保留第五次，本版本收尾。不自动补轮数、扫描更多学习率、改模型或补训练话术。一次相同seed实验不能给出跨seed稳定性结论。

## VS Code 入口

“终端 → 运行任务 → 核验第六次学习率实验（不训练）”仅核验。
用户决定开始后，选择“第六次训练（仅学习率0.0001，5轮，全新初始化）”；该任务先核验再启动，日志在专用终端。
**Ctrl+Shift+B的默认任务仍是第五次，不用它启动第六次。** 当前未执行 --train。

沿用引擎为每次启动新建 runs/时间-full-UUID，exist_ok=False；checkpoint/config/experiment/status/metrics均只在本次目录，不覆盖旧结果。不要同时开两次训练；VS Code任务限制同任务一实例。新结果和评分使用独立第六次目录，历史表暂不改写。

## 冻结机器清单

以下JSON用于启动门控，修改绑定文件需要明确的新版本，不可悄悄放宽校验。

```json
{
  "status": "frozen_not_started",
  "baseline_run": "runs/20261005T132346-full-ebf624ab40734b858b46553b4da37537",
  "selection": "artifacts/dev-check-expanded-v1/selection.json",
  "baseline_replies": "outputs/dev-expanded-fifth-20261005/replies.jsonl",
  "baseline_ratings": "outputs/repetition-review-20261006-v1/ai-expert-ratings-repetition-revised.json",
  "baseline_metrics": {
    "n": 135,
    "intent_mean": 1.8074074074074074,
    "help_mean": 1.325925925925926,
    "fluency_mean": 1.6888888888888889,
    "repetition": 63,
    "risk": 43
  },
  "baseline_best_dev_loss": 1.1029298457139332,
  "generation_baseline": {
    "checkpoint": "runs\\20261005T132346-full-ebf624ab40734b858b46553b4da37537\\best.pt",
    "checkpoint_sha256": "2a2e549f954d96325daf4cb5c286edae9f9441b2ea806bab0bcdd1734fa46479",
    "cpu_threads": 2,
    "decoding": "greedy; block PAD/BOS/USER/ASSISTANT",
    "device": "cpu",
    "max_new_tokens": 320,
    "model_config": {
      "assistant_token_id": 3,
      "bos_token_id": 1,
      "d_ff": 512,
      "d_model": 128,
      "dropout": 0.1,
      "eos_token_id": 4,
      "max_length": 512,
      "n_heads": 4,
      "n_layers": 4,
      "pad_token_id": 0,
      "tokenizer_artifact": "artifacts/tokenizer/bitext-bpe-v1",
      "tokenizer_revision": "43b80bbb10695d80aa442fa8435eb3ca8c33111554bf1d076006dcf2b7d76ea7",
      "user_token_id": 2,
      "vocab_size": 4000
    },
    "scope": "fixed dev diagnostic, not sealed_test",
    "seed": 42,
    "selection_sha256": "a43e6a0a377aae541dd3a6deb8ed6fa61fffc19a4a168b89527ce440fbb51fbe"
  },
  "dependencies_sha256": {
    "configs/training-sixth.json": "50b35f1c9acd46afc36e6b10b1509176a17328269581db73a3bc8d12fac16e2c",
    "configs/training-fifth-final.json": "f151956dae46208d6a4fc15e4a5ef7ce65bd8d76505545b7eb1a0696a90b5dfc",
    ".vscode/tasks.json": "a5cba90f614e82e8fcc2ebf63cd1e4255fdcf1094f9f4625ca62a6f564c4b446",
    "tests/test_training_sixth.py": "01a5267c3212cc5648337896e4ef15c7da23f1e93b966769023e6b4ed1a21957",
    "src/my_project_1/training_sixth.py": "4c217338f14800277e363227890ceb17a8db9a375ead0f6e4b967b90d8e874f3",
    "src/my_project_1/training.py": "8395cc47c728529976db69022ceeddfa27a1c60ca27fd00db8f0ffc6595f092a",
    "src/my_project_1/model.py": "ff34b43de9f32c864f04f5425f30d887a9bba399d915c80b5e3ad205c2c4544f",
    "src/my_project_1/attention.py": "73d7a38686fb39a829617a02b87403f9eb805f9f7eb6544cc09df97b8d121213",
    "src/my_project_1/transformer.py": "4a3224f3b1577b1ffa3819b187ebd87a9087c62a42522f3d69f8c7ef51ff95b2",
    "src/my_project_1/dataset.py": "54c63c50fd85fa3af480a4557502ef35bc8ff048ae3d36683f1b8a0b54f21ced",
    "src/my_project_1/tokenization.py": "9af3671c45504dfba09be278f1111a97324eb92742a5907cbac55a24e4383546",
    "src/utils/device.py": "2b7471eaa3c926e3a0c1cfc0150e016253172a182ee0900084557e06ab5e267b",
    "src/my_project_1/revisions.py": "cdbd05e82383975c066815f2b01972b9905a60fcb02f7a4656723326307a82b9",
    "src/my_project_1/target_revision_v31.py": "d8417d1fe459a3a6415a903611910d4f88067646a5c900ab81ad8ab92ed53fa3",
    "src/my_project_1/condition_measurement_v2.py": "a1ae29881cc8a752633e7178bf3fd4a9a211ce6fa7a0d5b1aad7ad5bb135ab0f",
    "src/my_project_1/generate.py": "97add8a57d069133b54d270d62e6161cd29f48d7267be7b949c73248eeb0695c",
    "configs/model.json": "1c0afa156f7afb403cfde51b1b01d12b48775f8d5a76a45719f3a102d0e870d0",
    "artifacts/tokenizer/bitext-bpe-v1/tokenizer.json": "43b80bbb10695d80aa442fa8435eb3ca8c33111554bf1d076006dcf2b7d76ea7",
    "artifacts/tokenizer/bitext-bpe-v1/config.json": "ac42fc8423c12ad1bca23370a9e55ae89aef89786c638f051160fd97cbdf2baf",
    "data/derived/refund-execution-v3.1/manifest.json": "5c281ecb802be3208bfcf0e0fa7d1a11f3a41a6ee79380e876f7f3c593f4729c",
    "data/derived/refund-execution-v3.1/train.jsonl": "4d1ed247e99b7b9aa5914be35ea9e6bcb8d4a8f70f1188808cda5b8bd07fbe73",
    "data/splits/bitext-v1/train.jsonl": "2a5186ebf69c691cb7848c6bd5509882158ff49477e1e99fdb3d0991df968396",
    "data/splits/bitext-v1/dev.jsonl": "35f7f1711e400290b1d20588382663d3a6c6c730c023a09470bec22c4de5d493",
    "artifacts/dev-check-expanded-v1/selection.json": "a43e6a0a377aae541dd3a6deb8ed6fa61fffc19a4a168b89527ce440fbb51fbe",
    "artifacts/dev-check-expanded-v1/freeze.json": "ca5da363b611d2a2cc1ff479f070f83e43db0c54d2d468e51c2c4c02a5f48d4c",
    "outputs/dev-expanded-fifth-20261005/replies.jsonl": "5d416b9e69bbe8be72d5f85c8c08723e90d23cf86e49f88190f6344cb5e5e7f5",
    "outputs/repetition-review-20261006-v1/ai-expert-ratings-repetition-revised.json": "bdf5ead98a4d6b5335de58abaf71a68df5389add296da0221399ae971b8dbf5c",
    "outputs/dev-expanded-fifth-20261005/config.json": "6ba6ab85ba9f8b983b71402b6539854fed9b550a0be34de8230acea1f6312df6",
    "outputs/repetition-review-20261006-v1/review-freeze.json": "a5a0da69026b75b994c6b58e0744926c28eea9dfb671f108ea98c6d83a376c77",
    "docs/evaluation-risk-rubric-v1.1.md": "3c7e4da6c11de7d67024ced8aa6c24f11c97a2f42a96320a7ddaa080b88bdea6",
    "docs/evaluation-rubric-v1.md": "b97dedda0490479fe4e25372a8abc594843e75728890b3f1ad2b0777ced0d68e",
    "runs/20261005T132346-full-ebf624ab40734b858b46553b4da37537/config.json": "86f0a44ea4a13cf2e3171be5c1a5843d09984e24c24d7688435dffaabdd39732",
    "runs/20261005T132346-full-ebf624ab40734b858b46553b4da37537/summary.json": "5d2f1a53c18351ae78357e9be9f9743919ef6f215be3d7c400be65d8abf6602e",
    "runs/20261005T132346-full-ebf624ab40734b858b46553b4da37537/status.json": "ff3e8610ebb56ac21bf1ec424202485a1ef8730393fadd18ff5bbacaf98088ce",
    "runs/20261005T132346-full-ebf624ab40734b858b46553b4da37537/experiment.json": "bc109889dc07e3554f5560bb807be29bdeb032dd154dcfbf4d3a50c2d0e4e66d",
    "runs/20261005T132346-full-ebf624ab40734b858b46553b4da37537/best.pt": "2a2e549f954d96325daf4cb5c286edae9f9441b2ea806bab0bcdd1734fa46479"
  }
}
```

## 方法验收

验收仅包含契约单测、模拟训练路由和真实只读预检；不会以“短训练一下”验证入口。真实训练及第六次质量比较尚未发生。核验记录追加唯一实验日志，不复制因素表或修改README。

