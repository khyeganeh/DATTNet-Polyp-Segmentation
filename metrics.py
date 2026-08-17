import torch


class SegmentationMetrics:

    def __init__(self, smooth=1e-6):
        self.smooth = smooth

    def __call__(self, pred, target):

        pred = torch.sigmoid(pred)
        pred = (pred > 0.5).float()

        target = target.float()

        tp = (pred * target).sum()

        fp = (pred * (1 - target)).sum()

        fn = ((1 - pred) * target).sum()

        tn = ((1 - pred) * (1 - target)).sum()

        dice = (
            2 * tp + self.smooth
        ) / (
            pred.sum() + target.sum() + self.smooth
        )

        iou = (
            tp + self.smooth
        ) / (
            tp + fp + fn + self.smooth
        )

        precision = (
            tp + self.smooth
        ) / (
            tp + fp + self.smooth
        )

        recall = (
            tp + self.smooth
        ) / (
            tp + fn + self.smooth
        )

        f1 = (
            2 * precision * recall
        ) / (
            precision + recall + self.smooth
        )

        accuracy = (
            tp + tn
        ) / (
            tp + tn + fp + fn + self.smooth
        )

        return {
            "dice": dice.item(),
            "iou": iou.item(),
            "precision": precision.item(),
            "recall": recall.item(),
            "f1": f1.item(),
            "accuracy": accuracy.item(),
        }