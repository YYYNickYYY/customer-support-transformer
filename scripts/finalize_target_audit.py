"""Compile explicit AI context/sample decisions and exact-string diagnostics."""
from collections import Counter
import hashlib
import json
from pathlib import Path

from audit_target_revision import OUT, ROOT, PROTOCOL, load, rows, save

# Authored after reading all260 before/after sentences and adjacent contexts.
# This is an explicit review exception list, not a grammar detector or risk scorer.
exceptions = {
    36: ('new_awkward_goal', 'work towards a resolution process将目标变成“解决流程”，较原句别扭；不是残句，需整句重写。'),
    70: ('new_subject_action_mismatch', 'This指保持耐心/配合这一做法；原句可使流程加快并确保结果，改后让This自己work towards，行动主体未明确。'),
    234: ('new_redundancy', 'working closely with you to work towards新增work语义复述，应以整句核查/协助表述代替局部替换。'),
    237: ('new_awkward_goal', '同36，work towards a resolution process是新引入的目标搭配问题。'),
    253: ('new_redundancy', 'while we work to work towards明显重复work，是此次局部替换新引入。'),
}
# Evidence confirmed from before/after and full selected responses.
residuals = {
    43: ('same_class_residual', 'previous', "Once I have these details, I'll ensure that your account is set up smoothly and securely.", '改隐私句后邻句仍确定安全开户，无实际安全流程证据；保守规则覆盖不足。'),
    187: ('same_class_residual', 'previous', 'This will ensure that we accurately identify your subscription and promptly unsubscribe you from our mailing list.', '隐私句已改，前句仍确保准确识别和退订成功，不是单纯服务意愿。'),
    204: ('out_of_scope_business_claim', 'next', 'Please note that this action is irreversible, and you may miss out on valuable updates and exclusive offers.', '取消保证已改；“不可逆”是原有政策/流程断言，不纳入同类v2。'),
}
# Fixed sample numbering is the saved JSON ordering, not content-selected.
sample_decisions = [
    ('normal_attitude', 'We prioritize protecting every piece of data you share with us.', '隐私重视/目的，不宣称已安全处理；保留反例。'),
    ('privacy_assurance', "your wife's information will be handled with the utmost care.", '第三人称被动处理保证未覆盖。'),
    ('no_target_guarantee', 'They will be more than happy to provide you with the necessary support and guidance.', '未见本类隐私/取消/解决保证；页面及权益待核实属其他问题。'),
    ('privacy_assurance', 'every piece of information you share will be handled with the utmost care.', '泛指信息+被动处理保证未覆盖。'),
    ('privacy_assurance', 'will handle it with the utmost care.', '代词it回指信息，原v1只接受固定your information宾语。'),
    ('edited_no_target_guarantee', 'we recommend sharing only the necessary information through an official support channel.', '隐私句完整、无新增残句；Thank you for contacting to us等为改写前已有缺陷，非v2范围。'),
    ('edited_no_target_guarantee', 'we recommend sharing only the necessary information through an official support channel.', '目的/态度及完整渠道建议保留。'),
    ('edited_no_target_guarantee', 'we recommend sharing only the necessary information through an official support channel.', '隐私保证已去除；priority service/withdrawal limits等账户权益是其他原生业务断言，不混入本次。'),
    ('edited_no_target_guarantee', 'we recommend sharing only the necessary information through an official support channel.', '信息指导为目的，隐私已改。'),
    ('edited_no_target_guarantee', 'we recommend sharing only the necessary information through an official support channel.', '服务体验目标及emoji不是同类结果保证。'),
    ('other_subscription_result', 'I can ensure that you receive all the latest updates, promotions, and news directly in your inbox.', '订阅/投递结果保证仍在；原v1仅改取消结果，v2是否扩展到订阅需显式定义，不能默默扩范围。'),
    ('no_target_guarantee', "Additionally, you may want to add our email address to your contacts or address book to help guarantee that our newsletter lands safely in your inbox.", '条件性导航/help guarantee目的建议，不按guarantee关键词认定确定处理保证；前句过滤预防断言另记业务核实。'),
    ('privacy_and_cancel_assurance', 'Your request will be promptly addressed to ensure your subscription is canceled.', '退订结果保证，并有信息高度保密保证，均未覆盖。'),
    ('cancel_assurance', "we'll ensure your unsubscribe request is executed promptly and accurately.", 'execute准确及时退订请求保证，非assist服务意愿。'),
    ('no_target_guarantee', 'We are here to help and ensure that you have a seamless experience.', '体验目标正常；确定注册表单/开始收邮件是不同业务问题。'),
    ('other_subscription_result', 'I will make sure to add you to our mailing list.', '隐私已改，但订阅成功话术原样保留；不是本v1取消规则漏执行。'),
    ('other_subscription_result', 'This will ensure that you receive all the latest updates, news, and exclusive content directly in your inbox.', '结果保证订阅投递，隐私句已改；不把全部业务断言混合修改。'),
    ('edited_no_target_guarantee', 'recommend sharing only the necessary information through an official support channel.', '隐私句已改；keep updated/take care为宽泛目的/服务意愿，未明确保证处理成功。'),
    ('cancel_assurance', 'This will ensure that we accurately identify your subscription and promptly unsubscribe you from our mailing list.', '改隐私句但邻句取消成功保证仍在；同260检查的187。'),
    ('no_target_guarantee', "You'll find a sign-up form where you can enter your email address and submit it to complete the process.", '未见本类取消/隐私/解决保证；具体表单及权益等业务能力不代表已核实，不在本干预范围。'),
]
candidate_examples = {
    6: ('privacy_assurance', '被动data处理保证。'),
    118: ('privacy_assurance', 'we will/we’ll handle your information后接and confidentiality，尾界限不符原v1精确规则。'),
    216: ('privacy_assurance', 'with utmost care缺the，原规则未涵盖。'),
    350: ('privacy_assurance', '直接guarantee securely处理信息。'),
    375: ('privacy_assurance', '直接guarantee保护data。'),
    480: ('cancel_assurance', 'personally+cancelled双l，原固定短语未覆盖。'),
    489: ('cancel_assurance', 'subscription is promptly canceled，无that这一版本原规则未覆盖。'),
    613: ('cancel_assurance', '邻句退订保证，存在于已经改隐私句的train。'),
    711: ('privacy_assurance', 'You can trust that替代we will起句，且and respect尾句不在原精确片段。'),
    95: ('normal_goal', 'Our aim+seamless experience，不是确定解决保证。'),
    299: ('normal_attitude', 'prioritize protecting，不能自动等同实际隐私处理结果保证。'),
    442: ('qualified_service_intent', 'do everything I can为努力协助；不因ensure自动标记保证。'),
    456: ('normal_experience', 'smooth cancellation experience属于v1.1体验目标，保留。'),
    530: ('request_care_not_privacy_handling', '处理请求的care，不等于处理隐私信息；邻句取消保证另记。'),
}


def main():
    freeze = load(OUT / 'selection.json')
    assert hashlib.sha256(PROTOCOL.read_bytes()).hexdigest() == freeze['protocol_sha256']
    for p, expected in freeze['source_hashes'].items():
        assert hashlib.sha256(Path(p).read_bytes()).hexdigest() == expected, p
    groups = load(OUT / 'changed-contexts.json')['groups']
    reviewed = []
    for g in groups:
        index = g['index']
        kind, notes = exceptions.get(index, ('no_new_seam_defect_observed', '完整改写句和前后句未观察到明确的新引入接缝问题；不是证明全文无风险/无原生语病。'))
        residual = None
        if index in residuals:
            rkind, field, quote, reason = residuals[index]
            assert quote in g['after'][field]
            residual = {'kind': rkind, 'context_field': field, 'quote': quote, 'notes': reason}
        reviewed.append({'context_index': index, 'members': g['members'], 'before': g['before'], 'after': g['after'],
                         'seam_decision': kind, 'notes': notes, 'additional_finding': residual, 'reviewer': 'AI上下文审查'})
    samples = load(OUT / 'selected-samples.json')
    assert len(samples) == len(sample_decisions) == 20
    samples_reviewed = []
    for n, (r, (kind, quote, notes)) in enumerate(zip(samples, sample_decisions), 1):
        assert quote in r['response'], r['id']
        samples_reviewed.append({'number': n, 'id': r['id'], 'intent': r['intent'], 'layers': r['layers'], 'kind': kind, 'quote': quote, 'notes': notes, 'reviewer': 'AI客服专家诊断，非风险评分'})
    candidates = load(OUT / 'candidate-contexts.json')['groups']
    anchors = []
    for index, (kind, notes) in candidate_examples.items():
        g = candidates[index - 1]
        assert index == g['index']
        anchors.append({'context_index': index, 'kind': kind, 'notes': notes, **{k: g[k] for k in ('context', 'members')}})
    train = rows(ROOT / 'train.jsonl')
    source = rows('data/splits/bitext-v1/train.jsonl')
    searches = ["we'll handle your information with the utmost care.", 'your privacy and will be handled with the utmost care.',
                'information will be handled with the utmost care', 'handle your information with utmost care',
                'while we work to work towards a resolution', 'any of any', 'account!I',
                "Are you receiving any error messages or facing any specific steps you're encountering any error messages?"]
    exact = []
    for quote in searches:
        matches = [r for r in train if quote.lower() in r['response'].lower()]
        originals = [r for r in source if quote.lower() in r['response'].lower()]
        exact.append({'query': quote, 'method': 'case-insensitive exact substring; not semantic retrieval', 'source_rows': len(originals), 'derived_rows': len(matches),
                      'derived_ids': [r['id'] for r in matches], 'intent_counts': dict(Counter(r['intent'] for r in matches)), 'evidence_rows': [{k: r[k] for k in ('id', 'intent', 'instruction', 'response')} for r in matches[:5]]})
    counts = dict(Counter(r['seam_decision'] for r in reviewed))
    save('ai-context-review.json', {'method': 'All260 edited sentences with before/after and immediate neighbours read. All727 lexical candidate sentences read; boundary/example neighbour contexts and20 frozen full responses reviewed. Candidate pool not all full-response manually scored. AI, known experiment, no merchant facts.',
         'rubric': 'risk-v1.1 boundaries used for diagnosis, NOT new dev ratings', 'changed_reviews': reviewed, 'seam_counts': counts,
         'sample_reviews': samples_reviewed, 'candidate_examples': anchors, 'exact_search': exact,
         'decision': 'Data coverage and local replacement seams are confirmed. Do not change lr/epochs/model; propose reviewed whole-sentence v2 in same guarantee scope, preserve v1. No training started; next run in VSCode by user.'})
    print(json.dumps({'seam_counts': counts, 'exact_search': [{k: r[k] for k in ('query', 'source_rows', 'derived_rows', 'intent_counts')} for r in exact], 'fixed_sample_categories': dict(Counter(r['kind'] for r in samples_reviewed))}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
