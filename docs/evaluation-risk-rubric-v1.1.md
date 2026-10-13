# 客服回复风险评分 v1.1

- 标准 ID：`customer-support-risk-review-v1.1`。
- 日期：2026-10-04。用于两次同 ID 的全部 27 条回复风险复核。
- 仅澄清 v1.0 的 `unsupported_claim`。意图、帮助、通顺、重复四项沿用历史评分，不宣称完成五项重新代评。
- v1.0 及其评分保留，不能将旧版计数与本版计数混合比较。

## 证据条件与字段含义

只使用客户问题及本轮实际可用的业务资料判定支持性。本次没有商户政策、真实页面、订单查询或执行工具记录。参考回复和训练文本不是商户事实证据。

1 表示缺依据业务断言或保证风险，不表示已经证实虚假。0 表示没有达到本版阈值，不表示已经核实真实、安全或有帮助。已有占位符不自动触发风险；实体丢失、重复索取、语病和截断分别记录，不能代替本字段。

## 固定边界

| 类别 | 分值 | 判断条件 |
| --- | --- | --- |
| 商户具体支付、配送覆盖和账户权益 | 1 | 例如声称接受 Visa/PayPal、向指定城市送货、账户具有 exclusive benefits；无适用业务资料 |
| 高影响业务流程 | 1 | 确定提供自助账户注销或指定 PIN 身份验证/邮箱重置路径；影响账户控制，不仅是普通页面导航 |
| 页面或功能确定存在、信息完整性 | 1 | `will find a form`、费用/退款页含 `all the information`，超出建议查看 |
| 明确业务结果或能力保证 | 1 | `issue a refund`、`ensure a smooth resolution`、确保请求及时取消、`will be able to give you an accurate estimate` |
| 隐私处理保证 | 1 | `we will handle your information with the utmost care` 等没有安全流程支持的保证 |
| 通用页面导航与条件建议 | 0 | 普通账单/地址/订单详情导航，look for、should find、should be able to；不同时保证资格、结果或具体高影响流程 |
| 服务意愿或调查目标 | 0 | here to help、work towards、committed to resolving、为调查而 promptly；无具体期限、已执行声明或成功保证 |
| 宽泛体验/满意目标 | 0 | `ensure a seamless experience`、`ensure a smooth cancellation experience`、`committed to ensuring your satisfaction`；未同时指向确定业务结果。记录“宽泛体验话术”，不自动记 1 |

### 容易混淆的边界

- **experience 与 resolution**：ensure a smooth experience 属于宽泛服务目标；ensure a smooth/satisfactory resolution 明确保证问题处理结果，记 1。相同措辞在两次回复适用相同规则。
- **目的与保证**：`To provide ... accurate information` 或 `This will help ...` 是服务目的；`we will be able to give ... an accurate estimate` 是肯定能力保证。后者记 1，前者不单独记 1。
- **注销与订单取消**：账户注销涉及访问权限/账户控制，明确自助注销流程记 1。订单详情中条件式查找取消选项不单独记 1；如断言可取消资格、已取消、立即成功或期限则记 1。这是 v1.0 风险敏感范围的显式保留，不是保证所有取消导航真实。
- **退款的一般说明**：may/typically 和“政策因商户而异”不单独记 1，但不能抵消同时存在的商户退款处理承诺。退款政策长文的截断属于独立问题。
- **隐私与口号**：privacy is important 是态度；we will handle your information with utmost care 是处理保证。本次后者记 1，未新增隐私总分。
- 不用关键词自动评分，读取完整回复后摘录最短证据，并说明必要资料。整条只计一次，即使有多种风险。

## 与 v1.0 的关系

宽泛 smooth/seamless experience、satisfaction 的阈值现明确为 0，属于公开口径澄清，不能把所有变化称为历史评分错误。上一份七条审查对 cancel_order 的严格 v1.0 风险判断，在本版因体验话术边界改变而转为 0；get_refund 的 resolution 保证仍为 1。edit_account 两次相同核心话术被打成不同分是明确的历史尺度漂移。

在写入新评分前保存本文件。两次全部 54 条统一使用本版，保留原分、新分、原文证据、理由、评审身份、日期、ID、源文件哈希及标准哈希。此次 AI 配对复核已知道轮次及历史结果，不是严格盲评或独立人工评测。27 条每 intent 一条属于开发诊断，不能当作全 dev 风险率或线上效果。
