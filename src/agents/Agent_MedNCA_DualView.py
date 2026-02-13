import einops
import torch
import torch.utils.data
import tqdm

from src.agents.Agent_MedNCA_Simple import MedNCAAgent
from src.utils.helper import merge_img_label_gt_simplified


class MedNCADualViewAgent(MedNCAAgent):
    """
    Agent for dual-view 2D segmentation.

    Expects dataset samples to provide:
      - image_a, label_a (BHWC)
      - image_b, label_b (BHWC)
    Optionally:
      - label_dist_a, label_dist_b (BHWC)
    """

    def prepare_data(self, data: dict, eval: bool = False) -> dict:
        def _as_tensor(x):
            return x if torch.is_tensor(x) else torch.as_tensor(x)

        img_a = _as_tensor(data["image_a"]).type(torch.FloatTensor).to(self.device)
        lbl_a = _as_tensor(data["label_a"]).type(torch.FloatTensor).to(self.device)
        img_b = _as_tensor(data["image_b"]).type(torch.FloatTensor).to(self.device)
        lbl_b = _as_tensor(data["label_b"]).type(torch.FloatTensor).to(self.device)

        # BHWC -> BCHW
        img_a = img_a.permute(0, 3, 1, 2).contiguous()
        lbl_a = lbl_a.permute(0, 3, 1, 2).contiguous()
        img_b = img_b.permute(0, 3, 1, 2).contiguous()
        lbl_b = lbl_b.permute(0, 3, 1, 2).contiguous()

        data["image_a"] = img_a
        data["label_a"] = lbl_a
        data["image_b"] = img_b
        data["label_b"] = lbl_b

        # Provide single-view-compatible keys for base training loop helpers.
        # These are used for logging/visualization only; the model consumes image_a/image_b.
        data["image"] = img_a
        data["label"] = lbl_a

        if "label_dist_a" in data and "label_dist_b" in data:
            dist_a = _as_tensor(data["label_dist_a"]).type(torch.FloatTensor).to(self.device)
            dist_b = _as_tensor(data["label_dist_b"]).type(torch.FloatTensor).to(self.device)
            data["label_dist"] = torch.cat([dist_a, dist_b], dim=0)

        return data

    def get_outputs(self, data: dict, full_img=True, **kwargs) -> dict:
        x_a, y_a = data["image_a"], data["label_a"]
        x_b, y_b = data["image_b"], data["label_b"]
        out = self.model(x_a, x_b, y_a, y_b, self.exp.config["trainer.batch_duplication"])
        return out

    @torch.no_grad()
    def test(
        self,
        scores,
        save_img: list = None,
        tag: str = "test/img/",
        pseudo_ensemble: bool = False,
        split="test",
        ood_augmentation=None,
        output_name: str = None,
        export_prediction: bool = False,
        prediction_export_path: str = "pred",
    ) -> dict:
        dataset = self.exp.datasets[split]
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=8, shuffle=False)
        self.exp.set_model_state("test")

        loss_log = {}
        if save_img is None:
            save_img = [1, 2, 3, 4, 5, 32, 45, 89, 357, 53, 122, 267, 97, 389]

        for i, data in enumerate(tqdm.tqdm(dataloader)):
            if ood_augmentation is not None:
                raise NotImplementedError("OOD augmentation not implemented for dual-view agent.")

            data = self.prepare_data(data, eval=True)
            pair_id = data["id"][0]

            if pseudo_ensemble:
                preds = []
                for k in range(10):
                    out_k = self.get_outputs(data, full_img=True, tag=str(k))
                    preds.append(out_k["logits"])
                stack = torch.stack(preds, dim=0)
                pred, _ = torch.median(stack, dim=0)
                del preds, stack
            else:
                pred = self.get_outputs(data, full_img=True, tag="0")["logits"]

            # pred: (2, H, W, C) because batch_size=1 pair.
            pred_a = pred[0:1]
            pred_b = pred[1:2]

            lbl_a = data["label_a"]  # BCHW
            lbl_b = data["label_b"]  # BCHW

            # Scores expect BHWC tensors.
            s_a = scores(
                pred=pred_a,
                target=einops.rearrange(lbl_a, "b c h w -> b h w c"),
                patient_id=f"{pair_id}_A",
            )
            s_b = scores(
                pred=pred_b,
                target=einops.rearrange(lbl_b, "b c h w -> b h w c"),
                patient_id=f"{pair_id}_B",
            )

            for key, val in s_a.items():
                loss_log.setdefault(key, {})
                loss_log[key][f"{pair_id}_A"] = val
            for key, val in s_b.items():
                loss_log.setdefault(key, {})
                loss_log[key][f"{pair_id}_B"] = val

            if i in save_img:
                img_a = einops.rearrange(data["image_a"].cpu(), "b c h w -> b h w c")
                img_b = einops.rearrange(data["image_b"].cpu(), "b c h w -> b h w c")

                pred_a_vis = pred_a.cpu()
                pred_b_vis = pred_b.cpu()
                gt_a_vis = einops.rearrange(lbl_a.cpu(), "b c h w -> b h w c")
                gt_b_vis = einops.rearrange(lbl_b.cpu(), "b c h w -> b h w c")

                self.exp.write_img(
                    f"{tag}{pair_id}_A_{i}",
                    merge_img_label_gt_simplified(img_a, pred_a_vis, gt_a_vis, rgb=dataset.is_rgb),
                    self.exp.currentStep,
                )
                self.exp.write_img(
                    f"{tag}{pair_id}_B_{i}",
                    merge_img_label_gt_simplified(img_b, pred_b_vis, gt_b_vis, rgb=dataset.is_rgb),
                    self.exp.currentStep,
                )

        self.exp.set_model_state("train")
        return loss_log
