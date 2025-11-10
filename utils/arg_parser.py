import argparse
import os
import yaml
from argparse import BooleanOptionalAction
from types import SimpleNamespace
from copy import deepcopy


class ArgsParser:
    def __init__(self, model_choices=("cyclegan", "animegan")):
        self.model_choices = model_choices

    def parse(self) -> SimpleNamespace:
        cli = self._build_cli().parse_args()

        config_path = os.path.join(cli.args_root, f"{cli.model}.yaml")
        cfg = self._load_yaml(config_path)

        cfg["model"] = cli.model

        cli_overrides = self._flatten_overrides(self._ns_to_dict(cli))
        cfg = self._deep_update(cfg, cli_overrides)

        self._validate_cfg(cfg)

        args = self._to_namespace(cfg)
        return args

    def _build_cli(self) -> argparse.ArgumentParser:
        p = argparse.ArgumentParser(
            "Train Anime/Cycle GAN with YAML config + CLI override")

        p.add_argument("--model", type=str, required=True,
                       choices=self.model_choices)
        p.add_argument("--args_root", type=str, required=True)

        # data
        p.add_argument("--photo_root", type=str, default=None)
        p.add_argument("--anime_style_root", type=str, default=None)
        p.add_argument("--anime_smooth_root", type=str, default=None)
        p.add_argument("--image_size", type=int, default=None)

        # train
        p.add_argument("--num_epochs", type=int, default=None)
        p.add_argument("--pretrain_epochs", type=int, default=None)
        p.add_argument("--batch_size", type=int, default=None)
        p.add_argument("--num_workers", type=int, default=None)
        p.add_argument("--save_every", type=int, default=None)
        p.add_argument("--out_dir", type=str, default=None)
        p.add_argument("--seed", type=int, default=None)
        p.add_argument("--resume", action=BooleanOptionalAction, default=False)
        p.add_argument("--start_epoch", type=int, default=None)
        p.add_argument("--ckpt_dir", type=str, default=None)
        p.add_argument("--decay_epoch", type=int, default=None)

        # optim
        p.add_argument("--g_lr_pretrain", type=float, default=None)
        p.add_argument("--g_lr", type=float, default=None)
        p.add_argument("--d_lr", type=float, default=None)

        # losses
        p.add_argument("--lambda_cyc", type=float, default=None)
        p.add_argument("--lambda_idt", type=float, default=None)
        p.add_argument("--lambda_adv_g", type=float, default=None)
        p.add_argument("--lambda_adv_d", type=float, default=None)
        p.add_argument("--lambda_con", type=float, default=None)
        p.add_argument("--lambda_gra", type=float, default=None)
        p.add_argument("--lambda_col", type=float, default=None)
        p.add_argument("--lambda_tv", type=float, default=None)
        p.add_argument("--lambda_smo", type=float, default=None)
        p.add_argument("--backbone", type=str, default=None)

        return p

    @staticmethod
    def _load_yaml(path: str) -> dict:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    @staticmethod
    def _deep_update(base: dict, override: dict) -> dict:
        base = deepcopy(base)
        for k, v in (override or {}).items():
            if isinstance(v, dict) and isinstance(base.get(k), dict):
                base[k] = ArgsParser._deep_update(base[k], v)
            else:
                base[k] = v
        return base

    @staticmethod
    def _ns_to_dict(ns: argparse.Namespace) -> dict:
        d = {}
        for k, v in vars(ns).items():
            if v is not None:
                d[k] = v
        return d

    @staticmethod
    def _flatten_overrides(cli: dict) -> dict:
        out = {"data": {}, "train": {}, "optim": {}, "losses": {}}

        # data
        for key in ["photo_root", "anime_style_root", "anime_smooth_root", "image_size"]:
            if key in cli:
                out["data"][key] = cli[key]

        # train
        for key in ["num_epochs", "batch_size", "num_workers", "save_every",
                    "out_dir", "seed", "resume", "start_epoch", "ckpt_dir", "decay_epoch"]:
            if key in cli:
                out["train"][key] = cli[key]

        # optim
        for key in ["g_lr", "d_lr"]:
            if key in cli:
                out["optim"][key] = cli[key]

        # losses
        for key in ["lambda_cyc", "lambda_idt", "lambda_adv_g", "lambda_adv_d",
                    "lambda_con", "lambda_gra", "lambda_col", "lambda_tv", "lambda_smo", "backbone"]:
            if key in cli:
                out["losses"][key] = cli[key]

        # model
        if "model" in cli:
            out["model"] = cli["model"]

        # loại bỏ block trống
        out = {k: v for k, v in out.items() if v not in ({}, None)}
        return out

    def _validate_cfg(self, cfg: dict):
        model = cfg.get("model")
        data = cfg.get("data", {})
        assert model in self.model_choices, f"model phải là {self.model_choices}"

        for k in ["photo_root", "anime_style_root"]:
            assert data.get(k), f"Thiếu data.{k} trong config"

        if model == "animegan":
            assert data.get(
                "anime_smooth_root"), "Thiếu data.anime_smooth_root cho AnimeGAN"

    @staticmethod
    def _to_namespace(cfg: dict) -> SimpleNamespace:
        flat = {}
        flat["model"] = cfg.get("model")

        data = cfg.get("data", {})
        flat["photo_root"] = data.get("photo_root")
        flat["anime_style_root"] = data.get("anime_style_root")
        flat["anime_smooth_root"] = data.get("anime_smooth_root")
        flat["image_size"] = data.get("image_size", 256)

        train = cfg.get("train", {})
        flat["num_epochs"] = train.get("num_epochs", 100)
        flat["pretrain_epochs"] = train.get("pretrain_epochs", 0)
        flat["batch_size"] = train.get("batch_size", 4)
        flat["num_workers"] = train.get("num_workers", 1)
        flat["save_every"] = train.get("save_every", 10)
        flat["out_dir"] = train.get("out_dir", "output")
        flat["seed"] = train.get("seed", 42)
        flat["resume"] = train.get("resume", False)
        flat["start_epoch"] = train.get("start_epoch", 0)
        flat["ckpt_dir"] = train.get("ckpt_dir")
        flat["decay_epoch"] = train.get("decay_epoch", 100)

        optim = cfg.get("optim", {})
        flat["g_lr_pretrain"] = optim.get("g_lr_pretrain")
        flat["g_lr"] = optim.get("g_lr")
        flat["d_lr"] = optim.get("d_lr")

        losses = cfg.get("losses", {})
        flat["lambda_cyc"] = losses.get("lambda_cyc")
        flat["lambda_idt"] = losses.get("lambda_idt")
        flat["lambda_adv_g"] = losses.get("lambda_adv_g")
        flat["lambda_adv_d"] = losses.get("lambda_adv_d")
        flat["lambda_con"] = losses.get("lambda_con")
        flat["lambda_gra"] = losses.get("lambda_gra")
        flat["lambda_col"] = losses.get("lambda_col")
        flat["lambda_tv"] = losses.get("lambda_tv")
        flat["lambda_smo"] = losses.get("lambda_smo")
        flat["backbone"] = losses.get("backbone")

        return SimpleNamespace(**flat)
