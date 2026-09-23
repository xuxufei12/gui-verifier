import json,tempfile,unittest,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from gui_verifier.experiment004 import parse_array,summarize,grouped

class ProtocolTests(unittest.TestCase):
 def test_strict_json_no_repairs(self):
  self.assertEqual(parse_array('["yes","unknown"]',2),['yes','unknown'])
  for text in ('```json\n["yes"]\n```','[true]','["YES"]','["yes","no"]','{"q":"yes"}'):
   self.assertIsNone(parse_array(text,1))
 def test_invalid_counts_as_errors(self):
  m=summarize([{'gold':['yes','no'],'predictions':None},{'gold':['unknown'],'predictions':['yes']}])
  self.assertEqual(m['strict_accuracy'],0);self.assertEqual(m['false_success_rate'],.5)
  self.assertEqual(m['confusion']['no']['invalid'],1)
 def test_group_neutral_ids_and_no_repeated_questions(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'data.jsonl';r={'id':'p1','group_id':'g','image':'i.png','request':{'goal':'g','rules':[],
    'questions':[{'id':'positive','text':'Visible?'}]},'labels':{'positive':'yes'}}
   p.write_text(json.dumps(r)+'\n'+json.dumps(r)+'\n')
   rows=grouped(p);self.assertEqual(len(rows),1);self.assertEqual(len(rows[0]['gold']),1)
   self.assertEqual(rows[0]['request']['questions'][0]['id'],'q01')
if __name__=='__main__':unittest.main()
