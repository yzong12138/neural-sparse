import torch


class SparseTrainingLoss:
    def __init__(self, weight=1):
        self.weight = weight

    def __call__(self, q_rep, d_rep, inputs, accelerator=None):
        raise NotImplementedError

    def get_loss(self, q_rep, d_rep, inputs, accelerator=None):
        return self.weight * self.__call__(
            q_rep, d_rep, inputs, accelerator=accelerator
        )


class KLDivLoss(SparseTrainingLoss):
    def __init__(self, in_batch_negatives=False, weight=1):
        self.in_batch_negatives = in_batch_negatives
        self.loss = torch.nn.KLDivLoss(reduction="none")
        super().__init__(weight)

    def __call__(self, q_rep, d_rep, inputs, accelerator=None):
        teacher_scores = inputs["scores"]

        if not self.in_batch_negatives:
            batch_size = q_rep.shape[0]
            d_rep = d_rep.reshape(batch_size, d_rep.shape[0] // batch_size, -1)
            student_scores = torch.bmm(
                d_rep, q_rep.reshape(batch_size, -1, 1)
            ).squeeze()

        student_scores = torch.log_softmax(student_scores, dim=1)
        teacher_scores = torch.softmax(teacher_scores, dim=1)
        loss = self.loss(student_scores, teacher_scores).sum(dim=1).mean(dim=0)
        return loss
