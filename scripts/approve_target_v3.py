"""Record finite AI sentence decisions; screened exclusions stay distinguishable."""
import re
from pathlib import Path
from my_project_1.expanded_dev import read, save
from my_project_1.tokenization import file_hash

ROOT=Path('outputs/target-revision-v3-review-20261005')
# Explicit non-duplicate-charge outcome/execution decisions after reading all603 candidates.
POSITIVE={15,26,78,123,153,174,186,207,221,253,297,308,323,334,352,354,372,374,393,432,502,510,517,559,574,599,669,688,713,726,753,763,764,773,818,854,868,875,885,886,939,957,968,1002,1005,1011,1012,1027,1045,1075,1081,1084,1095,1102,1107,1133,1192,1228,1253,1264,1265,1288,1291,1320,1326,1333,1369,1402,1409,1424,1454,1463,1480,1484,1509,1522,1542,1571,1587,1612,1648,1660,1665,1672,1694,1710,1734,1741,1759,1769,1789,1795,1797,1823,1842,1859,1877,1891,1910,1925,1929,1936,1947,1948,1966,1972,1987,2032,2105,2122,2165,2184,2195,2202,2221,2222,2230,2239,2256,2269,2275,2374,2408,2414,2431,2432,2436,2443,2460,2472,2514,2524,2541,2560,2564,2570,2596,2603,2609,2619,2624,2662,2716,2737,2773,2814,2834,2839,2891}
SAFE='We can help you review the issue and any refund request through an official support channel.'

def replacement(s):
    # Preserve headings and conditional factual scenario, not merchant outcome assertions.
    heading=re.match(r'^(\d+\.\s+(?:\*\*[^*]+\*\*:?[ \t]*|[^:]+:[ \t]*))',s)
    if not heading:heading=re.match(r'^\d+\.[ \t]+',s)
    prefix=heading.group(0) if heading else ''
    body=s[len(prefix):]
    condition=re.search(r'\b(?:If|In case|In the (?:unfortunate )?event (?:that|of)|Should)\b.+?(?=,\s*(?:please|we\b|our\b|a refund|you will|rest assured|it is|kindly|reach|notify|inform))',body,re.I)
    if not condition:condition=re.search(r'\b(?:If|In case|In the (?:unfortunate )?event (?:that|of)|Should)\b.+?(?=,)',body,re.I)
    if condition:
        c=condition.group().strip()
        return prefix+c[0].upper()+c[1:]+', please contact an official support channel to review the issue and any refund request.'
    if body.startswith('Once '):
        m=re.match(r'Once .+?,',body)
        if m:return prefix+m.group()+' we can help you review the request and the next steps through an official support channel.'
    return prefix+SAFE

def main():
    p=read(ROOT/'proposals.json');t=read(ROOT/'triage.json');reviews=[]
    for r in t['candidates']:
        n,s=r['number'],r['before']
        duplicate=bool(re.search(r'Duplicate|Double Charging|Duplicate or Overpayment|Overcharged or Double Billed|Billing Errors',s,re.I))
        yes=(duplicate and n not in {1203,2130}) or n in POSITIVE
        after=replacement(s) if yes else None
        # Mixed entities/questions require explicit preservation, not silent dropping.
        if yes and ('{{' in s or '?' in s):raise ValueError('Needs specific entity/question decision '+str(n))
        reviews.append({'number':n,'id':r['id'],'before':s,'after':after,'approved':yes,
            'reason':'明确退款执行/补救/付款结果断言；整句改为核查申请，保留条件与标题' if yes else '非确定退款执行/到账结果：服务意愿、引导申请、查询准确性或其他范围，保留'})
    payload={'reviewer':'AI finite full-sentence review, not independent human; lexical exclusions separately logged, no claim all-training semantic coverage',
        'proposals_sha256':file_hash(ROOT/'proposals.json'),'triage_sha256':file_hash(ROOT/'triage.json'),
        'reviews':reviews,'screened_excluded':t['excluded'],'rules':[{'id':r['id'],'intent':'*','before':r['before'],'after':r['after'],'kind':'refund-execution-whole-sentence'} for r in reviews if r['approved']]}
    save(ROOT/'approval.json',payload)
    print('approved',len(payload['rules']),'reviewed',len(reviews),'screen excluded',len(t['excluded']))

if __name__=='__main__':main()
