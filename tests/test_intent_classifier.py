import unittest
import numpy as np

from tools.intent_classifier import IntentClassifier, Config


class _DummyModel:
    """Simula um modelo Keras: sempre devolve [0.1, 0.9] para duas classes."""

    def predict(self, X):
        return np.array([[0.1, 0.9]])


class IntentClassifierTest(unittest.TestCase):
    """Testes unitários autocontidos para o IntentClassifier."""

    @classmethod
    def setUpClass(cls):
        # Config minimalista com duas intenções
        cfg = Config(dataset_name="dummy", codes=["foo", "bar"])
        cls.clf = IntentClassifier(config=cfg, load_model=None, examples_file=None)
        # Injeta modelo falso
        cls.clf.model = _DummyModel()

    # -------------------------------------------------------
    # Predição básica
    # -------------------------------------------------------
    def test_top_intent(self):
        top_intent, _ = self.clf.predict("exemplo qualquer")
        self.assertEqual(top_intent, "bar")

    def test_probability_dict(self):
        _, probs = self.clf.predict("outro exemplo")
        self.assertIsInstance(probs, dict)
        self.assertSetEqual(set(probs.keys()), {"foo", "bar"})
        self.assertAlmostEqual(probs["foo"], 0.1, places=6)
        self.assertAlmostEqual(probs["bar"], 0.9, places=6)

    # -------------------------------------------------------
    # One-hot encoder configurado corretamente
    # -------------------------------------------------------
    def test_one_hot_encoder(self):
        enc = self.clf.onehot_encoder
        foo_vec = enc.transform([["foo"]]).toarray()[0]
        bar_vec = enc.transform([["bar"]]).toarray()[0]
        np.testing.assert_array_equal(foo_vec, np.array([1, 0]))
        np.testing.assert_array_equal(bar_vec, np.array([0, 1]))


if __name__ == "__main__":       # Permite `python test_intent_classifier.py`
    unittest.main()
