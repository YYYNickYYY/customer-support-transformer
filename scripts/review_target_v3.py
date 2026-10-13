"""Conservative triage; finite candidates require subsequent explicit review."""
import re
from my_project_1.expanded_dev import read, save
from pathlib import Path

ROOT=Path('outputs/target-revision-v3-review-20261005')
QUAL=re.compile(r'\b(?:goals?|aim\w*|striv\w*|try\w*|potential\w*|may|might)\b|\b(?:do|doing)\b.{0,30}\b(?:best|everything|possible)\b|\b(?:work\w*|assist\w*|help\w*)\s+(?:towards|to|you|with|in)\b|\bmake every effort\b',re.I)
EXEC=re.compile(r'\b(?:process|issue|provide|arrange|initiate|credit)\s+(?:you\s+(?:with\s+)?|for\s+you\s+)?(?:a|the|your|an|any|appropriate)\s+(?:refund|reimbursement|rebate)\b|\brefund\s+(?:the|your|any)\s+(?:duplicate|extra|excess)\b|\brectif\w*\b',re.I)
PASSIVE=re.compile(r'\b(?:refund|reimbursement|rebate)\b.{0,45}\bwill\s+be\s+(?:processed|issued|credited|completed|provided)\b|\byou\s+will\s+receive\s+(?:a|the|your|an)\s+(?:refund|reimbursement|rebate)\b|\b(?:ensure|guarantee)\b.{0,45}\b(?:receive|refund|reimbursement|rebate)\b',re.I)

def triage(s):
    if QUAL.search(s):return 'qualified-service-or-possibility-retained'
    if PASSIVE.search(s):return None
    if EXEC.search(s) and re.search(r'\b(?:we|I|they|team|support)\b.{0,100}\b(?:will|can|able)\b',s,re.I):return None
    return 'no-explicit-refund-execution-predicate-conservative-exclusion'

def main():
    p=read(ROOT/'proposals.json');c=[];x=[]
    for i,r in enumerate(p['rules'],1):
        reason=triage(r['before'])
        if reason:x.append({'number':i,'id':r['id'],'before':r['before'],'reason':reason})
        else:c.append({'number':i,**r})
    save(ROOT/'triage.json',{'candidates':c,'excluded':x,'method':'conservative lexical triage, exclusions not claimed fully AI reviewed; candidates require explicit whole sentence/context review; scope coverage not all semantic risks'})
    print('candidates',len(c),'excluded',len(x))

if __name__=='__main__':main()
