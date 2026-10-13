"""Record AI-reviewed finite sentence decisions; never mutate proposals or v1."""
import json
import re
from collections import Counter

from my_project_1.target_revision import json_bytes, digest
from my_project_1.target_revision_v2 import PROPOSALS, SAFE
from my_project_1.tokenization import file_hash

# Explicit resolution decisions after reading every unique candidate sentence.
RESOLUTION = {37, 62, 68, 91, 92, 94, 112, 114, 132, 142, 144, 147, 151, 191,
192, 197, 218, 225, 240, 256, 257, 260, 262, 271, 283, 284, 297, 304, 313,
330, 342, 365, 366, 381, 389, 403, 411, 416, 425, 429, 450, 462, 464, 466,
467, 480, 499, 502, 523, 531, 540, 546, 575, 578, 599, 609, 613, 615, 623,
631, 646, 667, 671, 680, 686, 699, 701, 713, 720, 746}
CANCEL = {66, 106, 118, 130, 156, 158, 183, 199, 222, 288, 321, 357, 376,
436, 454, 484, 503, 512, 528, 532, 561, 563, 579, 620, 626, 629, 637,
670, 681, 689, 696, 700, 712}
# High precision predicate-object support for reviewed privacy sentences;
# excludes request-care, billing/refund workflows and future importance attitudes.
OBJECT = r'(?:information|data|details|email(?: address)?|privacy)'
PRIVACY_PREDICATE = re.compile(
    rf'\b{OBJECT}\b[^.!?]{{0,70}}\bwill\s+(?:be\s+(?:handl\w*|treat\w*|kept|secur\w*|protect\w*|safeguard\w*|stor\w*)|remain\s+(?:confidential|secure|safe))\b'
    rf'|\b(?:will|we\x27ll)\s+(?:carefully\s+)?(?:handle|treat|keep|protect|safeguard|secure|store)\b[^.!?]{{0,60}}\b(?:{OBJECT}|it|them)\b'
    rf'|\b(?:will|we\x27ll)\s+(?:take every measure|take every precaution|take all necessary measures|take the necessary measures|take every step)[^.!?]{{0,90}}\b{OBJECT}\b'
    rf'|\bguarantee\w*\b[^.!?]{{0,100}}\b{OBJECT}\b', re.I)


def main():
    p = json.loads((PROPOSALS / 'proposals.json').read_text(encoding='utf-8'))
    approved, rejected, overrides = [], [], {}
    for n, r in enumerate(p['rules'], 1):
        before, kind = r['before'], r['kind']
        if kind == 'privacy-whole-sentence':
            yes = bool(PRIVACY_PREDICATE.search(before)) and not re.search(r'\b(?:payment|refund|rebate|password|access key|PIN|account closure|account deletion|deleted|updated|update)\b|\{\{', before, re.I)
            if re.search(r'\bhandle\s+(?:it|them)\b', before, re.I) and not re.search(r'\b(?:information|data|details)\b', before, re.I):
                yes = False  # Ambiguous request antecedent: do not infer personal data.
            if re.search(r'\bhandle\s+(?:your|this|the)\s+request\b', before, re.I):
                yes = False  # Request-care is not an information-handling guarantee.
            reason = 'AI核查：明确隐私信息未来处理保证；密码/请求care/其他业务及重要性态度不纳入' if yes else '排除：非明确隐私处理谓词，或混有其他业务动作/实体，有限规则不强行整句替换'
        elif kind == 'resolution-whole-sentence':
            yes = n in RESOLUTION
            reason = 'AI核查：直接保证解决结果，无努力限定，无需删除具体询问信息/业务入口' if yes else '排除：目的/客户配合/努力限定，或混合路由/步骤需另行核查'
        elif kind == 'cancel-whole-sentence':
            yes = n in CANCEL
            reason = 'AI核查：直接保证订阅取消/请求执行结果' if yes else '排除：订阅成功非原三类，或定位/处理能力混合句暂不扩大范围'
        else:
            yes, reason = True, 'AI核查：诊断证实的5处接缝或安全开户邻句'
        if yes:
            approved.append(r['id'])
            # Preserve the specific original email inquiry in this mixed sentence.
            if n == 637:
                overrides[r['id']] = "Alternatively, you can let me know your email address, and I can help you review the cancellation steps."
        else:
            rejected.append({'id': r['id'], 'before': before, 'reason': reason})
        r['review_reason'] = reason
    before = 'This will ensure that we accurately identify your subscription and promptly unsubscribe you from our mailing list.'
    extra = {'id': 'reviewed-' + digest(before.encode())[:16], 'intent': 'newsletter_subscription', 'before': before,
             'after': 'This information will help us review your cancellation request and suggest the next steps.', 'kind': 'cancel-context-completeness'}
    payload = {'review_date': '2026-10-05', 'reviewer': 'AI客服专家整句核查，不是独立人工代评',
        'proposals_sha256': file_hash(PROPOSALS / 'proposals.json'), 'approved_rule_ids': approved,
        'after_overrides': overrides, 'extra_rules': [extra], 'rejected': rejected,
        'reviews': [{k: r[k] for k in ('id', 'before', 'after', 'kind', 'review_reason')} for r in p['rules']],
        'scope': 'finite confirmed resolution, newsletter cancellation, privacy handling sentences; other business claims remain out of scope'}
    destination = PROPOSALS / 'approval.json'
    if destination.exists():
        raise ValueError('Approval already exists; refuse overwrite')
    destination.write_bytes(json_bytes(payload))
    print(json.dumps({'approved_rules': len(approved) + 1, 'rejected_rules': len(rejected),
                      'kinds': dict(Counter(r['kind'] for r in p['rules'] if r['id'] in approved))}, ensure_ascii=False))


if __name__ == '__main__':
    main()
