"""Persist explicit full-text audit findings; never change ratings or targets."""
from pathlib import Path
from my_project_1.expanded_dev import read, save
from my_project_1.tokenization import file_hash
from audit_fifth_side_effects import OUT

TARGETS = {
'2143':'地址修改步骤/官方协助合理，无相同步骤或句子重复。',
'2779':'地址编辑正常提示；double-check为核对动作而非同义叠词，未改。',
'2340':'常规地址步骤完整，不重复；与第五次for accuracy叠加不同。',
'4238':'原目标已将Person Name替换成Salutation/Last Name，实体缺陷原有；追问范围相关。',
'3993':'12588替换为占位符原有；笼统追问，但没有重复索已给发票号。',
'4235':'原有人名实体替换；追问范围/账户相关，无明显重复。',
'7723':'投诉指导及后续记录/处理有无依据断言，未改；感谢前后功能不同，不机械算重复。',
'7633':'do my utmost努力限定，正常服务意愿，保留；无明显句子循环。',
'7765':'投诉详情及证据追问，goal限定服务愿望，保留。',
'8988':'人工转接泛问具体事项，无真实转接；不是循环重复。',
'9257':'确定Chat Now入口及营业/能力未核实，未改；普通帮助口号不同作用。',
'9521':'模拟正在协调真人转接，无转接工具；不属于本次退款改写。',
'10741':'问题确实是给未婚夫开户，目标亲属有依据；模型另造father不能据此免责。',
'10416':'开户必要信息、官方渠道与体验目标，无相同体验短语反复。',
'10756':'客户确实为女儿开户且失败，澄清有效；不是自造亲属。',
'15292':'目标姓名替换、拟真检索/安全发送能力未核实，原有；不是新改写问题。',
'15795':'12588丢失与not mistaken语病原有；条件下载建议不等于第五次确定下载结果。',
'15404':'mail场景询发送邮箱相关；姓名替换原有，无发票词循环。',
'16362':'未改且ensuring you receive the refund明确结果保证；前面的assist不能覆盖保证分句。',
'16575':'退款日期/支付方式/交易号澄清有效，guidance并非执行保证；Currency实体变化原有。',
'15901':'订单号澄清有效，guide through refund process是引导，不保证批准到账。',
'15993':'第五次批准保证句替换正确，必要订单/交易号及工作目标保留，无新增重复；金额实体变化原有。',
'16833':"成功退款保证已替换为核查下一步；开头I'm with you语病原有，无新接缝。",
'16197':'保证获得退款已改核查；work towards结尾是努力目标，不应删；必要标识保留。',
'17291':'订阅状态无工具却声明会查，超本次因素；无明显循环。',
'17184':'take care of canceling subscription promptly仍有执行退订保证，非本次退款因素；开头语病原有。',
'17085':'退款外退订操作能力表述与低效索理由仍有，未改；正常官方渠道提示保留。',
'18383':'付款报错澄清有效，work towards目标合理，无相同目的双述。',
'18644':'付款问题官方支持路径，正常步骤，无重复付款体验短语。',
'18144':'付款故障澄清及共情合理，do everything possible限定协助，非成功保证。',
'21506':'报告注册问题详情澄清合理，无相邻重复赞许。',
'21077':'注册问题及未来体验目标合理，无相邻proactive复述。',
'21362':'赞许、细节调查、服务改进与感谢作用不同，未见生成式重复proactive句。',
'22648':'反馈详情澄清有效，确保到达部门断言未核实；没有form or form。',
'22821':'目标本来就重复反馈到达部门目的，属于原有冗余；也含未核实确定部门送达。',
'22117':'确定表单存在断言原有，但候选没有form or form重复。',
'24666':'gold实体丢失、账户升级/独占权益保证原有；体验只一次，无重复转换目标。',
'24400':'使用账户指南较长，24/7支持/权限能力未经核实；未出现生成的重复转换语句。',
'24017':'常规换档步骤、未核实权限；首尾服务目标不同形式但无相邻同义转换重复。',
'24985':'给定订单号保留，但模拟查询能力未核实；不再索订单号。',
'25416':'已给订单号后询商户并非重复索同一标识；主体需求保留，未查实业务状态。',
'25707':'原目标已出现Order Number连续两次，原生实体重复存在，非第五次新增。',
'26033':'未改目标说actively working on processing compensation且模拟查询，实质处理/能力断言仍有。',
'26611':'目标确定状态专页存在，未核实，非到账保证；给具体查询路径。',
'26211':'working diligently to process request为努力限定，保留；I get it the importance语病及模拟查询原有。',
'26516':'批准后末尾保证到账已正确替换，但前句ensure everything is in order仍保证业务状况，旧句残留，不是新接缝。',
'26285':'批准句正确核查替换；name/email泛问不如退款标识明确，但应另因素处理，不混改。',
'26576':'批准保证正确替换；专页状态功能断言原有，超本次因素；无新语病/重复。',
}
CASES = {
'10191':('fragment','account for your account病态拼接；训练精确片段0命中，不足证明纯模型因果。'),
'10438':('service_goal','重复体验目标且自造father；单个体验短语可在未改目标找到，重复组合未证实来自目标。'),
'15563':('fragment','invoices related to invoices循环；精确片段0命中，原目标实体错误与生成冗余分开。'),
'16355':('refund_result','满意/高效resolution保证仍在；satisfactory完整片段0命中，efficient片段8命中但未改，可能重组，不能单一归因。'),
'17634':('fragment','email newsletter词句循环，完整循环片段0命中，非上限截断。'),
'17906':('service_goal','理解问题的澄清目的重复，而第四是调查动作；未改目标普通理解句不能证明重复源自改写。'),
'18587':('service_goal','首尾seamless payment experience重复，付款目标未改。'),
'18680':('same_candidate','error message or any error messages重复候选，精确片段0命中。'),
'21649':('service_goal','首尾完全同感谢语；短句476命中且3条改，但短句存在不是证明重复排列来自改写。'),
'21669':('service_goal','相邻proactive赞许复述，注册目标未改。'),
'22079':('same_candidate','form or form相同候选，完整片段0命中。'),
'22284':('same_candidate','form or form相同候选，完整片段0命中。'),
'22756':('service_goal','分享想法/感谢前后复述为敏感边界；第四也有价值及分享句，应纳入统一重复尺度复核。'),
'24393':('same_candidate','seamless and seamless及smooth transition连续重复，换档目标未改，所查完整叠词0命中。'),
'24407':('service_goal','seamless transition/experience相邻同义目标，换档目标未改。'),
'24479':('service_goal','smooth/seamless transition相邻复述，完整片段0命中。'),
'24647':('service_goal','smooth transition and ensure smooth transition字面重复，完整片段0命中。'),
'25673':('known_identifier','已给Order Number又索取，查单目标原生实体重复存在，但不能直接证明模型学到同一重复模式。'),
'26320':('refund_result','与退款目标R003/R005近似ensure smooth resolution，且非上限；目标残留与生成重组均可能。'),
'26438':('refund_result','退款进度保证、泛问曾退款；有同类残留目标，不自动归为单纯生成问题。'),
'2946':('service_goal','double-check accuracy ... for accuracy重复，完整片段0命中，地址目标未改。'),
'4364':('known_identifier','已给85632又索发票号，编号先替换后索取；非上限，实体缺陷与无谓交互分开。'),
'7011':('service_goal','前后认真处理投诉复述；第四自身复述投诉也接近边界，统一复核不可只对新输出更敏感。'),
'7162':('service_goal','necessary guidance在相邻句复述，投诉目标未改。'),
'7590':('service_goal','两个指导申请的服务承诺复述，投诉目标未改。'),
'8916':('scoring_disagreement','第四和第五都有To connect...details...better understand相同核心复述；0→1不能据此认定新出现重复，需同标准新快照统一复核。'),
'9158':('fragment','live chat/agent长句循环且电话用作聊天地址；所查重复片段0命中。'),
}
RESIDUAL = {
'R001':('result_guarantee','Receiving reimbursement...ensure smooth resolution结合上下文保证结果，无努力程度限定；保守执行谓词排除漏掉。'),
'R002':('result_guarantee','最后ensuring that you receive appropriate compensation为明确获得保证；help字样不授权补偿，未改遗漏。'),
'R003':('result_guarantee','协助输入后ensure smooth resolution是结果保证；accurate查询及无工具模拟查询另问题。'),
'R004':('result_guarantee','退款语境work together...ensure smooth resolution未限定尽力，按相同v1.1结果边界判遗漏。'),
'R005':('result_guarantee','cooperation will ensure smooth resolution是直接肯定处理结果，未改遗漏。'),
'R006':('boundary_process_goal','help us ensure smooth resolution process接近流程体验目标，暂列边界复核，不机械判遗漏或删。'),
'R007':('qualified_service','do everything we can努力限定，正常服务意愿保留，不因ensure关键词修改。'),
}

def main():
    frozen=read(OUT/'material-freeze.json')
    for p,h in read(OUT/'selection.json')['dependencies_sha256'].items():assert file_hash(p)==h
    for n,h in frozen.items():assert file_hash(OUT/n)==h
    targets=read(OUT/'targets.json');cases=read(OUT/'cases.json');contexts=read(OUT/'residual-contexts.json')
    assert {r['id'].split(':')[-1] for r in targets}==set(TARGETS)
    assert {r['id'].split(':')[-1] for r in cases}==set(CASES)
    assert {r['review_id'] for r in contexts}==set(RESIDUAL)
    triage=read('outputs/target-revision-v3-review-20261005/triage.json')
    approval=read('outputs/target-revision-v3-review-20261005/approval-frozen.json')
    source_texts=[r['v2'] for r in targets+contexts]
    linked=[{**x,'subsequent_review':next((r for r in approval['reviews'] if r['id']==x['id']),None)} for x in triage['excluded']
        if any(x['before'] in text for text in source_texts)]
    save(OUT/'review.json',{'reviewer':'AI full-text diagnostic; not independent human; original scores untouched',
        'material_hashes':frozen,'target_reviews':[{'id':r['id'],'intent':r['intent'],'layers':r['layers'],'changed':r['changed'],'finding':TARGETS[r['id'].split(':')[-1]]} for r in targets],
        'case_reviews':[{'id':r['id'],'intent':r['intent'],'cohort':r['cohort'],'category':CASES[r['id'].split(':')[-1]][0],'finding':CASES[r['id'].split(':')[-1]][1]} for r in cases],
        'residual_reviews':[{'review_id':r['review_id'],'ids':r['ids'],'classification':RESIDUAL[r['review_id']][0],'finding':RESIDUAL[r['review_id']][1]} for r in contexts],
        'triage_exclusions':linked,'triage_source_sha256':file_hash('outputs/target-revision-v3-review-20261005/triage.json'),
        'approval_source_sha256':file_hash('outputs/target-revision-v3-review-20261005/approval-frozen.json'),
        'limitations':'Exact spans not causal proof. 48 stratified diagnostic samples not population defect rates. At least one repeat score disagreement requires uniform fourth/fifth review; no silent historical overwrite.',
        'review_script_sha256':file_hash(__file__)})
    save(OUT/'review-freeze.json',{'review_sha256':file_hash(OUT/'review.json'),'script_sha256':file_hash(__file__),'no_training_or_target_changes':True})
    print('Reviewed 27 cases, 48 full targets, 7 residual contexts; preserved historical scores.')

if __name__=='__main__':main()
