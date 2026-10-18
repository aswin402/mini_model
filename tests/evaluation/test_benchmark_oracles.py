from types import SimpleNamespace

from little.evaluation.comprehensive_matrix import SOTABenchmarkSuite


class ConstantKernel:
    def __init__(self, reply):
        self.reply = reply
    def chat(self, prompt):
        return self.reply
    def process(self, prompt):
        return SimpleNamespace(inference=SimpleNamespace(status='SUPPORTED', answer=self.reply), tool_result=None, code_program=None, learning=None)


def fake_suite(reply):
    suite = SOTABenchmarkSuite.__new__(SOTABenchmarkSuite)
    suite.kernel = ConstantKernel(reply)
    suite.rust = SimpleNamespace(is_accelerated=False)
    return suite


def test_constant_three_does_not_pass_all_math(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = fake_suite('3').run_all()
    math_items = [item for item in result['items'] if item['modality'] == 'Math & CAS Reasoning']
    assert sum(item['passed'] for item in math_items) <= 1
    assert len(math_items) == 7


def test_noisy_unknown_is_not_refutation(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = fake_suite('I do not know; yes, no evidence available.').run_all()
    refuted = [item for item in result['items'] if item['expected_status'] == 'REFUTED']
    # During transition the old labels also identify these cases.
    refuted += [item for item in result['items'] if item['expected_status'] == 'refuted']
    assert len(refuted) == 2
    assert not any(item['passed'] for item in refuted)
