import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score

class Evaluator:
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device

    def evaluate_accuracy(self, dataloader):
        self.model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                preds = outputs.argmax(dim=1)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())
        return accuracy_score(y_true, y_pred)

    def evaluate_auc(self, dataloader, pos_class=2):
        self.model.eval()
        y_true, y_scores = [], []
        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = F.softmax(self.model(images), dim=1)
                y_scores.extend(outputs[:, pos_class].cpu().numpy())
                y_true.extend((labels == pos_class).cpu().numpy())
        return roc_auc_score(y_true, y_scores)