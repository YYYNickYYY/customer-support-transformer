"""Persist explicit full-text review decisions and bounded lexical diagnostics."""
import json
from pathlib import Path
from my_project_1.expanded_dev import read, save
from my_project_1.tokenization import file_hash

OUT=Path('outputs/dev-expanded-20261005-v1')
NOTES={
'17370':'未来隐私处理保证已替换；goal/satisfaction为服务目标，保留合理。',
'16930':'退订结果保证已整句替换为核查步骤和官方申请。',
'17863':'退订结果保证已整句替换；getting in touch to us语病原文已有。',
'17755':'现时隐私处理断言与订阅权益仍在，超出冻结未来式隐私及退订范围。',
'17346':'订阅资讯和权益断言仍在，不是退订结果保证，超出冻结范围。',
'17399':'make every effort是努力限定的服务意愿，不能仅凭resolve判结果保证。',
'17248':'邮件退订路径、尊重隐私及aim为一般引导/目标，没有明确退订成功保证。',
'15927':'适当结果保证已去除；Currency Symbol到$的实体变化原先已有。',
'15973':'结果保证已去除，核查原因和材料请求保留。',
'16708':'positive resolution保证已去除；not mistaken your confusion语病原文已有。',
'16043':'票据、购买日期、金额及条款的核查，没有明确承诺付款。',
'16029':'do everything we can限定努力，按v1.1保留服务意愿，不判自动退款保证。',
'16425':'work towards是服务意愿，不保证退款成功。',
'16447':'结果保证已在v1处理，v2保留work towards，未发现v2漏执行。',
'6792':'结果保证已去除；support will have access和最新状态能力断言残留，超出旧范围。',
'6484':'positive resolution已去除；we will promptly rectify situation仍承诺实际补救，非旧ensure/guarantee/promise子类。',
'6358':'may/usually/work towards/assist限定，未发现确定退款保证。',
'6313':'process a refund if necessary仍断言退款执行能力；条件不证明模型有执行权限，超出旧范围。',
'5983':'generally 7–14 days不是确定时效保证；支持访问账户及实时更新能力仍有未经核实断言。',
'6321':'process reimbursement if necessary仍属退款执行能力断言，超出旧范围。',
'13888':'估计日期和索取追踪号为核查意愿，没有准确性结果保证。',
'13596':'assist determining estimated date为服务意愿，不判准确保证。',
'13405':'we can give you a precise estimate是精确查询能力断言，超出v2冻结范围。',
'13731':'we will be able to give you an accurate estimate是准确估计能力断言，超出v2范围。',
'25375':'完全保密保证已去除；准确提供信息作为目的，不单独判确定准确性保证。',
'25637':'保密保证已去除；access your order details和will promptly update仍有查单能力断言。',
'25110':'隐私安全保证已去除；访问订单以提供最准确信息仍有能力断言。',
'25257':'通用自助查询导航，没有明确结果保证；真实入口未核实。',
'25415':'订单号重复原文已有；通用查询引导，不是v2新接缝问题。',
'25044':'准确性作为目标，协助追踪/估计，不单独判准确结果保证。',
'25416':'询问公司以指导客户，准确作为目的/服务目标，不判结果保证。',
'26444':'结果保证已去除；Allow me to review和While I dive into this仍模拟实际查退款，超出旧范围。',
'26329':'v1 work to work接缝已修复；专门状态页面必然存在的断言仍在，超出旧范围。',
'26268':'smooth resolution保证已去除；调查/全面更新作为目的，不单独判实际查询已完成。',
'25903':'you will be able to view status/recent updates仍有页面功能断言，超出旧范围。',
'26136':'Let me swiftly dive into current status模拟实际查询；MoneyAmount实体错误和无效追问原文已有。',
'26545':'Allow me quickly check/Just give me moment模拟实际查询；MoneyAmount实体问题原文已有。',
'26072':'should find为条件性导航；committed to providing为服务意愿，不能判保证。',
}

def main():
    freeze=read(OUT/'target-audit/selection.json')
    for p,h in freeze['dependencies_sha256'].items():assert file_hash(p)==h
    targets=read(OUT/'target-audit/targets.json')
    assert {r['id'].split(':')[-1] for r in targets}==set(NOTES)
    reviews=[{'id':r['id'],'intent':r['intent'],'layers':r['layers'],
        'v2_changed_from_v1':r['v2']!=r['v1'],'finding':NOTES[r['id'].split(':')[-1]],
        'source': 'targets.json contains complete original/v1/v2 for this ID'} for r in targets]
    train=[json.loads(l) for l in Path('data/derived/result-privacy-v2/train.jsonl').read_text(encoding='utf-8').splitlines()]
    ratings=read(OUT/'ai-expert-ratings.json')['ratings']
    selected_intents={r['intent'] for r in targets}
    diagnostics=[]
    for model in ('second','third','fourth'):
        generated=[json.loads(l) for l in (OUT/model/'replies.jsonl').read_text(encoding='utf-8').splitlines()]
        by={r['id']:r for r in ratings if r['model']==model}
        for r in generated:
            if r['intent'] not in selected_intents or not by[r['id']]['unsupported_claim']:continue
            for quote in by[r['id']]['evidence']:
                assert quote in r['generated_response']
                if len(quote.split())<5:continue
                matches=[t['id'] for t in train if quote.lower() in t['response'].lower()]
                diagnostics.append({'model':model,'id':r['id'],'intent':r['intent'],'quote':quote,
                    'v2_exact_substring_count':len(matches),'matching_train_ids':matches})
    save(OUT/'target-audit/review.json',{'method':'AI complete original/v1/v2 comparison, frozen stratified 44 positions / 38 unique rows; no merchant facts verified; not population risk rates',
        'selection_sha256':file_hash(OUT/'target-audit/selection.json'),'targets_sha256':file_hash(OUT/'target-audit/targets.json'),'reviews':reviews})
    save(OUT/'target-audit/generated-source-diagnostics.json',{'method':'Case-insensitive exact substring search in ALL v2 training responses for >=5-word evidence excerpts from risk-marked replies in six audited intents. Zero matches means this span is not verbatim present, not proof of model-only cause; matches do not prove memorization or causation. Selected excerpts may include repetition evidence; not risk classifier.',
        'train_sha256':file_hash('data/derived/result-privacy-v2/train.jsonl'),'diagnostics':diagnostics})
    print(json.dumps({'reviewed':len(reviews),'excerpt_diagnostics':len(diagnostics),'no_verbatim_match':sum(r['v2_exact_substring_count']==0 for r in diagnostics)}))

if __name__=='__main__':main()
