import wandb
import torch
import random
from tqdm import tqdm
import torch.optim as optim
import pytorch_lightning as pl
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from point_e.models.download import load_checkpoint
from point_e.util.plotting import render_point_cloud
from point_e.diffusion.sampler import PointCloudSampler
from point_e.models.configs import MODEL_CONFIGS, model_from_config
from shapetalk import PROMPTS, SOURCE_LATENTS, TARGET_LATENTS, NUM_POINTS
from point_e.diffusion.configs import DIFFUSION_CONFIGS, diffusion_from_config

VAL = "val"
TEST = "test"
TEXTS = "texts"
OUTPUT = "output"
SOURCE = "source"
TARGET = "target"
UPSAMPLE = "upsample"
MODEL_NAME = "base40M-textvec"


class SPICE(pl.LightningModule):
    def __init__(
        self,
        lr: float,
        batch_size: int,
        copy_prob: float,
        copy_prompt: str,
        dev: torch.device,
        cond_drop_prob: float,
        val_dataloader: DataLoader,
        test_dataloader: DataLoader,
    ):
        super().__init__()
        self.lr = lr
        self.dev = dev
        self.copy_prob = copy_prob
        self.batch_size = batch_size
        self.copy_prompt = copy_prompt
        self._init_model(cond_drop_prob)
        self._init_data(val_dataloader, VAL)
        self._init_data(test_dataloader, TEST)

    def _init_model(self, cond_drop_prob):
        self.diffusion = diffusion_from_config(DIFFUSION_CONFIGS[MODEL_NAME])
        upsampler_diffusion = diffusion_from_config(DIFFUSION_CONFIGS[UPSAMPLE])
        config = MODEL_CONFIGS[MODEL_NAME]
        config["cond_drop_prob"] = cond_drop_prob
        self.model = model_from_config(config, self.dev)
        self.model.load_state_dict(load_checkpoint(MODEL_NAME, self.dev))
        self.model.create_control_layers()
        upsampler_model = model_from_config(MODEL_CONFIGS[UPSAMPLE], self.dev)
        upsampler_model.eval()
        upsampler_model.load_state_dict(load_checkpoint(UPSAMPLE, self.dev))
        self.sampler = PointCloudSampler(
            device=self.dev,
            guidance_scale=[3.0, 0.0],
            aux_channels=["R", "G", "B"],
            model_kwargs_key_filter=(TEXTS, ""),
            models=[self.model, upsampler_model],
            num_points=[NUM_POINTS, 4096 - NUM_POINTS],
            diffusions=[self.diffusion, upsampler_diffusion],
        )

    def _init_data(self, dataloader, split):
        if dataloader is None:
            return
        assert len(dataloader) == 1
        batch = next(iter(dataloader))
        source_key, target_key = SOURCE + "_" + split, TARGET + "_" + split
        log_data = {source_key: [], target_key: []}
        prompts, source_latents, target_latents = (
            batch[PROMPTS],
            batch[SOURCE_LATENTS],
            batch[TARGET_LATENTS],
        )
        for prompt, source_latent, target_latent in tqdm(
            zip(prompts, source_latents, target_latents),
            total=len(prompts),
            desc=f"Initating {split} data",
        ):
            source_samples = self.sampler.sample_batch(
                batch_size=1,
                model_kwargs={},
                prev_samples=source_latent.unsqueeze(0),
            )
            target_samples = self.sampler.sample_batch(
                batch_size=1,
                model_kwargs={},
                prev_samples=target_latent.unsqueeze(0),
            )
            log_data[source_key].append(self._plot(source_samples, prompt))
            log_data[target_key].append(self._plot(target_samples, prompt))
            log_data[source_key].append(self._plot(source_samples, self.copy_prompt))
            log_data[target_key].append(self._plot(source_samples, self.copy_prompt))
        wandb.log(log_data, step=None)

    def _plot(self, samples, prompt):
        pc = self.sampler.output_to_point_clouds(samples)[0]
        fig = render_point_cloud(pc)
        img = wandb.Image(fig, caption=prompt)
        plt.close()
        return img

    def _sample_t(self):
        return (
            torch.tensor(
                random.sample(range(len(self.diffusion.betas)), self.batch_size)
            )
            .to(self.dev)
            .detach()
        )

    def configure_optimizers(self):
        return optim.Adam((self.parameters()), lr=self.lr)

    def training_step(self, batch, batch_idx):
        prompts, source_latents, target_latents = (
            batch[PROMPTS],
            batch[SOURCE_LATENTS],
            batch[TARGET_LATENTS],
        )
        if random.random() < self.copy_prob:
            x_start = source_latents
            texts = [self.copy_prompt] * len(prompts)
        else:
            x_start = target_latents
            texts = prompts
        terms = self.diffusion.training_losses(
            x_start=x_start,
            model=self.model,
            t=self._sample_t(),
            model_kwargs={TEXTS: texts, "guidance": source_latents},
        )
        loss = terms["loss"].mean()
        wandb.log({"loss": loss.item()}, step=None)
        return loss

    def _sample(self, prompt, guidance):
        return self.sampler.sample_batch(
            batch_size=1,
            guidances=[guidance, None],
            model_kwargs={TEXTS: [prompt]},
        )

    def validation_step(self, batch, batch_idx):
        self._val_test_step(batch, batch_idx, OUTPUT + "_" + VAL)

    def test_step(self, batch, batch_idx):
        self._val_test_step(batch, batch_idx, OUTPUT + "_" + TEST)

    def _val_test_step(self, batch, batch_idx, key):
        assert batch_idx == 0
        outputs = []
        prompts, source_latents = batch[PROMPTS], batch[SOURCE_LATENTS]
        with torch.no_grad():
            for prompt, source_latent in zip(prompts, source_latents):
                samples = self._sample(prompt, source_latent)
                outputs.append(self._plot(samples, prompt))
                samples = self._sample(self.copy_prompt, source_latent)
                outputs.append(self._plot(samples, self.copy_prompt))
        wandb.log({key: outputs}, step=None)
