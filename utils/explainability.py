"""
Simple explainability utilities:
- Attention-based text highlights
- Structured feature contribution via numeric gradient approximation
"""
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"

class SimpleExplainer:
    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.vocab = pickle.load(open(DATA_DIR / "vocab.pkl", "rb"))
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.feature_meta = pickle.load(open(DATA_DIR / "feature_names.pkl", "rb"))

    def _get_feature_names(self):
        names = []
        names += self.feature_meta.get("numeric", [])
        # OneHotEncoder categories
        encoder = pickle.load(open(DATA_DIR / "encoder.pkl", "rb"))
        for i, cat in enumerate(self.feature_meta.get("cat", [])):
            for val in encoder.categories_[i]:
                names.append(f"{cat}={val}")
        names += self.feature_meta.get("binary", [])
        return names

    @torch.no_grad()
    def explain_text(self, text_indices, text_attn):
        """
        text_indices: (seq_len,) or (1, seq_len)
        text_attn: (seq_len,) attention weights from model
        Returns top contributing words.
        """
        if text_indices.dim() == 2:
            text_indices = text_indices.squeeze(0)
        if text_attn.dim() == 2:
            text_attn = text_attn.squeeze(0)
        # mask out padding
        tokens = []
        weights = []
        for idx, w in zip(text_indices.cpu().tolist(), text_attn.cpu().tolist()):
            if idx == 0:
                continue
            word = self.inv_vocab.get(idx, "<unk>")
            tokens.append(word)
            weights.append(w)
        # Sort by weight
        sorted_idx = np.argsort(weights)[::-1]
        top_words = [(tokens[i], float(weights[i])) for i in sorted_idx[:5]]
        return top_words

    @torch.no_grad()
    def explain_structured(self, structured_vec, text_indices, denial_logit, missed_logit, delta=0.01):
        """
        Approximate feature importance by perturbing each feature slightly.
        structured_vec: (1, struct_dim)
        text_indices: (1, seq_len) actual text tokens
        """
        base_denial = torch.sigmoid(denial_logit).item()
        base_missed = torch.sigmoid(missed_logit).item()
        importances = []
        for i in range(structured_vec.shape[1]):
            perturbed = structured_vec.clone()
            perturbed[0, i] += delta
            d_logit, m_logit, _ = self.model(perturbed, text_indices)
            new_denial = torch.sigmoid(d_logit).item()
            new_missed = torch.sigmoid(m_logit).item()
            denial_diff = new_denial - base_denial
            missed_diff = new_missed - base_missed
            importances.append((i, denial_diff, missed_diff))
        feature_names = self._get_feature_names()
        # Sort by absolute denial impact
        importances.sort(key=lambda x: abs(x[1]), reverse=True)
        top_features = []
        for idx, d_imp, m_imp in importances[:5]:
            name = feature_names[idx] if idx < len(feature_names) else f"feat_{idx}"
            top_features.append({
                "feature": name,
                "denial_impact": round(float(d_imp), 4),
                "missed_revenue_impact": round(float(m_imp), 4),
            })
        return top_features

    def explain(self, structured_vec, text_indices, denial_logit, missed_logit, text_attn):
        """Full explanation bundle."""
        return {
            "top_text_tokens": self.explain_text(text_indices, text_attn),
            "top_structured_features": self.explain_structured(structured_vec, text_indices, denial_logit, missed_logit),
            "denial_probability": round(float(torch.sigmoid(denial_logit).item()), 4),
            "missed_revenue_probability": round(float(torch.sigmoid(missed_logit).item()), 4),
        }
