import os
from pathlib import Path
import sys
import unittest
import numpy as np
import yaml
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from dotenv import load_dotenv

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.intent_classifier import IntentClassifier, Config


class _DummyModel:
    """Simula um modelo Keras: sempre devolve [0.1, 0.9] para duas classes."""

    def predict(self, X):
        n = len(X)
        return np.tile(np.array([[0.1, 0.9]]), (n, 1))


class IntentClassifierTest(unittest.TestCase):
    """Testes unitários autocontidos para o IntentClassifier."""

    def setUp(self):
        print(f"\n🧪 Running {self._testMethodName}...")

    from dotenv import load_dotenv

    @classmethod
    def setUpClass(cls):
        load_dotenv()
        env_url = os.getenv("WANDB_MODEL_URL")
        if not env_url:
            raise unittest.SkipTest(
                "WANDB_MODEL_URL not set. Please set it in your .env file. "
                "You can find the model URL in the W&B interface under "
                "'Artifacts' > 'Model' > 'Usage'."
            )
        print("\n🌐 WANDB_MODEL_URL detected, loading real model...")
        cls.clf = IntentClassifier()
        print("✅ Model loaded from WandB")

    # -------------------------------------------------------
    # Predição básica
    # -------------------------------------------------------
    def test_top_intent(self):
        print("🔎 Checking top intent prediction")
        top_intent, _ = self.clf.predict("exemplo qualquer")
        print(f"Predicted intent: {top_intent}")
        self.assertIsInstance(top_intent, str)

    def test_probability_dict(self):
        print("📈 Checking probability dictionary")
        _, probs = self.clf.predict("outro exemplo")
        print(f"Probabilities: {probs}")
        self.assertIsInstance(probs, dict)
        self.assertGreaterEqual(len(probs), 1)

    # -------------------------------------------------------
    # One-hot encoder configurado corretamente
    # -------------------------------------------------------
    def test_one_hot_encoder(self):
        print("🔢 Validating one-hot encoder")
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
        print("✅ Model correctly loaded from WANDB")
        self.assertIsNotNone(self.clf.model)

    def test_model_accuracy_easy_examples(self):
        url = os.getenv("WANDB_MODEL_URL")
        if url:
            print("🌐 Using model from WANDB for accuracy check")
        else:
            print("⚙️ Using dummy model for accuracy demonstration")

        
        examples_path = os.path.join(os.path.dirname(__file__), "..", "tools", "confusion", "dataset-teste.csv")

        # ✅ Carrega o CSV
        df = pd.read_csv(examples_path)
        print(f"📂 Loaded examples from {examples_path} (CSV)")
        
        # ✅ Seleciona no máximo 10 amostras
        samples = df.sample(n=min(10, len(df)), random_state=42)
        texts = samples["texto"].tolist()
        labels = samples["tipo"].tolist()

        # examples_path = os.path.join(os.path.dirname(__file__), "..", "tools", "confusion", "confusion_examples.yml")
        # with open(examples_path, "r") as f:
        #     data = yaml.safe_load(f)

        # print(f"📂 Loaded examples from {examples_path}")
        # samples = []
        # for intent_block in data:
        #     for text in intent_block["examples"]:
        #         samples.append((text, intent_block["intent"]))
        #         if len(samples) >= 10:
        #             break
        #     if len(samples) >= 10:
        #         break

        # texts = [t for t, _ in samples]
        # labels = [l for _, l in samples]
        preds = self.clf.predict(texts)
        pred_labels = [p[0] for p in preds]

        for text, pred, label in zip(texts, pred_labels, labels):
            print(f"🗣️ Texto: {text}")
            print(f"➡️ Predição: {pred} | 🎯 Esperado: {label}")
            print("---")

        accuracy = sum(p == l for p, l in zip(pred_labels, labels)) / len(labels)
        print(f"🏆 Sample accuracy: {accuracy:.2f}")

        # Classification report and confusion matrix
        report = classification_report(labels, pred_labels, zero_division=0)
        print("\n📄 Classification Report:\n" + report)

        all_labels = sorted(set(labels) | set(pred_labels))
        cm = confusion_matrix(labels, pred_labels, labels=all_labels)
        cm_df = pd.DataFrame(cm, index=all_labels, columns=all_labels)
        print("\n🔢 Confusion Matrix:\n" + cm_df.to_string())

        if url:
            self.assertGreaterEqual(accuracy, 0.5)
        else:
            print("ℹ️ WANDB_MODEL_URL not set - skipping accuracy assertion")


if __name__ == "__main__":       # Permite `python test_intent_classifier.py`
    unittest.main()
