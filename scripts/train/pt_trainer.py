import logging
import torch


from transformers import Trainer, AutoModelForMaskedLM

logger = logging.getLogger(__name__)


class PTTrainer(Trainer):
    def __init__(self, model_args, data_args, **kwargs):
        self.model_args = model_args
        self.data_args = data_args
        self.ranking_loss_moving_avg = 0
        super().__init__(**kwargs)
        self.loss = torch.nn.KLDivLoss(reduction="none")
        self.set_teacher(model_args.model_name_or_path)

    def compute_loss(self, model, inputs, return_outputs=False):
        with torch.no_grad():
            teacher_logits = self.teacher(**inputs)[0]
            mask = inputs["attention_mask"].unsqueeze(-1)
            masked_hidden = teacher_logits.masked_fill(mask == 0, float('-inf'))
            teacher_logits = torch.max(masked_hidden, dim=1)[0]
            # teacher_logits = torch.max(
            #     teacher_logits * inputs["attention_mask"].unsqueeze(-1), dim=1
            # )[0]

        output = model(**inputs)[0][:, 0, :]
        assert teacher_logits.shape == output.shape

        student_scores = torch.log_softmax(output, dim=1)
        teacher_scores = torch.softmax(teacher_logits, dim=1)
        loss = self.loss(student_scores, teacher_scores).sum(dim=1).mean(dim=0)
        
        loss += (output.mean() - teacher_logits.mean())**2

        self.ranking_loss_moving_avg = (
            0.01 * loss.item() + 0.99 * self.ranking_loss_moving_avg
        )
        
        if self.state.global_step % self.args.logging_steps == 0:
            logger.info(
                f"Step {self.state.global_step}. ranking loss moving avg:{self.ranking_loss_moving_avg}"
            )
            logger.info(f"{teacher_logits[0][1234:1240]} {output[0][1234:1240]}")
            logger.info(f"{(loss, output) if return_outputs else loss}")
        return (loss, output) if return_outputs else loss

    def set_teacher(self, model_id):
        teacher = AutoModelForMaskedLM.from_pretrained(model_id)
        self._move_model_to_device(teacher, self.args.device)
        wrap_teacher = self._wrap_model(teacher, training=False)
        use_accelerator_prepare = True if teacher is wrap_teacher else False
        if use_accelerator_prepare:
            self.accelerator.prepare(teacher)
            self.teacher = teacher
        else:
            self.teacher = wrap_teacher
