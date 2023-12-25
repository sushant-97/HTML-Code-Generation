from rouge_score import rouge_scorer

class Accuracy:
    "A simple Accuracy function compatible with HF models"
    def __init__(self):
        self.count = 0
        self.tp = 0.
    def update(self, logits, labels):
        logits, labels = logits.argmax(dim=-1).view(-1).cpu(), labels.view(-1).cpu()
        tp = (logits == labels).sum()
        self.count += len(logits)
        self.tp += tp
        return tp / len(logits)
    def compute(self):
        return self.tp / self.count
    


def compute_rouge(predictions, references):
    # Initialize ROUGE scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    # Calculate scores for each prediction-reference pair
    scores = [scorer.score(ref, pred) for pred, ref in zip(predictions, references)]
    
    # Aggregate scores for all predictions
    aggregated_scores = {
        'rouge1': sum(score['rouge1'].fmeasure for score in scores) / len(scores),
        'rouge2': sum(score['rouge2'].fmeasure for score in scores) / len(scores),
        'rougeL': sum(score['rougeL'].fmeasure for score in scores) / len(scores)
    }
    return aggregated_scores