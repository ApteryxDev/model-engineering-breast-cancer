import ast
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def settings(source, name):
    tree = ast.parse(source)
    return [{kw.arg: ast.literal_eval(kw.value) for kw in call.keywords}
            for call in ast.walk(tree) if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name) and call.func.id == name]


class ModelSettingsTests(unittest.TestCase):
    def test_notebook_matches_script(self):
        script = (ROOT / "src/compare_models.py").read_text()
        nb = json.loads((ROOT / "notebooks/generate_report_figures.ipynb").read_text())
        code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
        for name, required in [
            ("StratifiedKFold", {"n_splits": 5, "shuffle": True, "random_state": 42}),
            ("RandomForestClassifier", {"n_estimators": 300, "random_state": 42, "n_jobs": -1}),
        ]:
            self.assertEqual(settings(script, name), [required])
            self.assertEqual(settings(code, name), [required])


if __name__ == "__main__":
    unittest.main()
