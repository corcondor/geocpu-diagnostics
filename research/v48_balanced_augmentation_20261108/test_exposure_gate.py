import sys
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parent))
from exposure_gate import gate

class ExposureGateTests(unittest.TestCase):
    def setUp(self):
        self.old={"supervised_tokens":1000,"source_supervised_tokens":{"original_parent":700,"python_docs_ja":300}}
        self.new={"supervised_tokens":1600,"source_supervised_tokens":{
            "original_parent":700,"python_docs_ja":300,
            "v48_python":150,"v48_logic":150,"v48_evidence":150,"v48_japanese":150}}

    def test_success_while_preserving_old_absolute_counts(self):
        r=gate(self.old,self.new)
        self.assertTrue(r["passed"])
        self.assertEqual(r["old_groups_compared"],2)

    def test_reduction_rejected_even_if_ratios_equal(self):
        self.new["source_supervised_tokens"]["original_parent"]=699
        self.assertFalse(gate(self.old,self.new)["passed"])

    def test_one_new_domain_missing_rejected(self):
        self.new["source_supervised_tokens"].pop("v48_japanese")
        self.assertFalse(gate(self.old,self.new)["passed"])

    def test_balancing_token_counts_not_row_counts(self):
        self.new["source_supervised_tokens"]["v48_python"]=600
        self.assertFalse(gate(self.old,self.new)["passed"])

    def test_same_total_budget_rejected(self):
        self.new["supervised_tokens"]=1000
        self.assertFalse(gate(self.old,self.new)["passed"])

    def test_tolerance_validation(self):
        with self.assertRaises(ValueError):
            gate(self.old,self.new,1)

if __name__=="__main__":
    unittest.main()
