import os
import sys
import unittest
import numpy as np
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.intent_classifier import IntentClassifier, Config


class _DummyModel:
    """Simula um modelo Keras: sempre devolve [0.1, 0.9] para duas classes."""

    def predict(self, X):
        return np.array([[0.1, 0.9]])


class IntentClassifierTest(unittest.TestCase):
    """Testes unitários autocontidos para o IntentClassifier."""

    @classmethod
    def setUpClass(cls):
        env_url = os.getenv("WANDB_MODEL_URL")
        if env_url:
            # Quando WANDB_MODEL_URL estiver definido, o IntentClassifier buscará o modelo automaticamente
            cls.clf = IntentClassifier()
        else:
            # Config minimalista com duas intenções usando modelo dummy
            cfg = Config(dataset_name="dummy", codes=["foo", "bar"])
            cls.clf = IntentClassifier(config=cfg, load_model=None, examples_file=None)
            cls.clf.model = _DummyModel()

    # -------------------------------------------------------
    # Predição básica
    # -------------------------------------------------------
    def test_top_intent(self):
        top_intent, _ = self.clf.predict("exemplo qualquer")
        if os.getenv("WANDB_MODEL_URL"):
            self.assertIsInstance(top_intent, str)
        else:
            self.assertEqual(top_intent, "bar")

    def test_probability_dict(self):
        _, probs = self.clf.predict("outro exemplo")
        self.assertIsInstance(probs, dict)
        if os.getenv("WANDB_MODEL_URL"):
            self.assertGreaterEqual(len(probs), 1)
        else:
            self.assertSetEqual(set(probs.keys()), {"foo", "bar"})
            self.assertAlmostEqual(probs["foo"], 0.1, places=6)
            self.assertAlmostEqual(probs["bar"], 0.9, places=6)

    # -------------------------------------------------------
    # One-hot encoder configurado corretamente
    # -------------------------------------------------------
    def test_one_hot_encoder(self):
        enc = self.clf.onehot_encoder
        codes = list(self.clf.codes)
        for idx, code in enumerate(codes):
            vec = enc.transform([[code]]).toarray()[0]
            self.assertEqual(len(vec), len(codes))
            # the vector should be one-hot with 1 at the correct index
            self.assertAlmostEqual(vec[idx], 1.0)
            self.assertTrue(((vec == 0) | (vec == 1)).all())
            decoded = enc.inverse_transform([vec])[0][0]
            self.assertEqual(decoded, code)

    def test_env_model_loaded(self):
        if os.getenv("WANDB_MODEL_URL"):
            self.assertIsNotNone(self.clf.model)
        else:
            self.skipTest("WANDB_MODEL_URL not set")

    def test_model_accuracy_easy_examples(self):
        url = os.getenv("WANDB_MODEL_URL")
        if not url:
            self.skipTest("WANDB_MODEL_URL not set")

        examples_path = os.path.join(os.path.dirname(__file__), "..", "tools", "confusion", "confusion_examples.yml")
        with open(examples_path, "r") as f:
            data = yaml.safe_load(f)

        samples = []
        for intent_block in data:
            for text in intent_block["examples"]:
                samples.append((text, intent_block["intent"]))
                if len(samples) >= 10:
                    break
            if len(samples) >= 10:
                break

        texts = [t for t, _ in samples]
        labels = [l for _, l in samples]
        preds = self.clf.predict(texts)
        pred_labels = [p[0] for p in preds]

        accuracy = sum(p == l for p, l in zip(pred_labels, labels)) / len(labels)
        self.assertGreaterEqual(accuracy, 0.5)


if __name__ == "__main__":       # Permite `python test_intent_classifier.py`
    unittest.main()
