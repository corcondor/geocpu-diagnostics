import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

def import_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f'{name}.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

class Smoke(unittest.TestCase):
    def test_train_dev_test(self):
        gen = import_script('build_curriculum')
        with tempfile.TemporaryDirectory() as d:
            report = gen.generate(Path(d))
            self.assertEqual([report['counts'][s]['rows'] for s in ('train','dev','test')],[640,160,160])
            self.assertEqual(report['counts']['train']['by_family']['evidence_chain'],128)
            seen=set()
            for split in ('train','dev','test'):
                for line in (Path(d)/f'v48_{split}.jsonl').read_text(encoding='utf-8').splitlines():
                    row=json.loads(line)
                    self.assertEqual(row['full_text'],row['prompt']+row['target'])
                    self.assertLessEqual(len(row['full_text']), 2048)
                    self.assertNotIn(row['prompt'],seen); seen.add(row['prompt'])
                    self.assertEqual(len(row['loss_mask_spans']),int(row['domain'] != 'evidence_chain'))
                    for a,b in row['loss_mask_spans']:
                        self.assertEqual(row['full_text'][a:b].splitlines()[0],'```output')
    def test_logger(self):
        l = import_script('colab_live')
        self.assertEqual(l.extract('Progress:  40/200 | Acc: 25.0%')['progress'],
                         {'done':40,'total':200,'reported_accuracy_percent':25.0})
        self.assertEqual(l.extract('PYTHON_TIR_STEP_30500_PROGRESS 10/120')['python_progress']['done'],10)
        with tempfile.TemporaryDirectory() as d:
            ret=l.execute(['python','-u','-c','print("Progress: 1/2 | Acc: 50.0%")'], Path(d)/'live', 1.0)
            self.assertEqual(ret,0)
            stat=json.loads((Path(d)/'live'/'status.json').read_text())
            self.assertEqual(stat['progress']['done'],1)
            self.assertEqual(stat['status'],'complete')
    def test_scorecards(self):
        c=import_script('compare_scorecards')
        blank={k:{} for k in c.TIERS}
        self.assertEqual(len(c.compare({'tiers':blank},{'tiers':blank})),0)
        a={k:{} for k in c.TIERS};b={k:{} for k in c.TIERS}
        a['T1']['accuracy']={'correct':10,'total':20,'cohort_sha256':'a','scorer_id':'exact1','decoding_id':'greedy'}
        b['T1']['accuracy']={**a['T1']['accuracy'],'correct':12}
        self.assertEqual(c.compare({'tiers':a},{'tiers':b})[0]['delta_percentage_points'],10.0)
        b['T1']['accuracy']['cohort_sha256']='other'
        self.assertIn('not_comparable',c.compare({'tiers':a},{'tiers':b})[0]['status'])
if __name__=='__main__':unittest.main()