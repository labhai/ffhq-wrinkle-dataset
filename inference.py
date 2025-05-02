import argparse
import logging
import os
from pathlib import Path
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from collections import OrderedDict
from tqdm import tqdm
from unet.unet_model import UNet
from unet.swin_unetr import SwinUNETR


def parse_args():
    parser = argparse.ArgumentParser(description="Wrinkle Segmentation")
    parser.add_argument(
        "--network",
        type=str,
        required=True,
        choices=["UNet", "SwinUNETR"],
        help="Network architecture: UNet or SwinUNETR",
    )
    parser.add_argument(
        "--num_channels",
        type=int,
        default=4,
        help="Number of input channels (Default: RGB+Texture=4)",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=2,
        help="Number of output classes (e.g. background+wrinkle=2)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./pretrained_ckpt/stage2_wrinkle_finetune_unet/stage2_unet.pth",
        required=True,
        help="Path to the trained model checkpoint (.pth)",
    )
    parser.add_argument(
        "--gpu_id",
        type=str,
        default="0",
        help="GPU ID to use (default: 0). If not available, CPU will be used",
    )
    parser.add_argument(
        "--image_path",
        type=str,
        required=True,
        help="Path to an image file or a directory of image files",
    )
    parser.add_argument(
        "--texture_path",
        type=str,
        required=True,
        help="Path to a texture file or a directory of texture files",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=1024,
        help="Image size for inference (default: 1024x1024)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./test_outputs",
        help="Output directory to save the predicted masks",
    )
    return parser.parse_args()


class WrinkleTestDataset(Dataset):
    """
    A dataset that combines RGB images and grayscale texture images into a 4-channel tensor.
    If both paths are directories, it matches files with the same names.
    If both paths are files, it forms a single (image, texture) pair.
    """

    def __init__(self, image_path, texture_path, img_size=None):
        super().__init__()
        self.img_size = img_size
        self.samples = []
        img_p = Path(image_path)
        tex_p = Path(texture_path)

        if img_p.is_dir() and tex_p.is_dir():
            self._collect_pairs_from_dirs(img_p, tex_p)
        elif img_p.is_file() and tex_p.is_file():
            if not img_p.exists():
                raise FileNotFoundError(f"Cannot find image file: {img_p}")
            if not tex_p.exists():
                raise FileNotFoundError(f"Cannot find texture file: {tex_p}")
            filename = img_p.name
            self.samples.append((str(img_p), str(tex_p), filename))
        else:
            raise ValueError(
                f"Both --image_path and --texture_path must be either directories or files.\n"
                f"Given: image_path={image_path}, texture_path={texture_path}"
            )

        if len(self.samples) == 0:
            raise RuntimeError("No valid (image, texture) pairs found.")

    def _collect_pairs_from_dirs(self, img_dir: Path, tex_dir: Path):
        valid_exts = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]
        img_files = [
            f
            for f in img_dir.iterdir()
            if f.is_file() and f.suffix.lower() in valid_exts
        ]
        img_files = sorted(img_files, key=lambda x: x.name)

        tex_dict = {}
        for root, dirs, files in os.walk(tex_dir):
            root_path = Path(root)
            for file in files:
                if Path(file).suffix.lower() in valid_exts:
                    tex_dict[file] = str(root_path / file)

        for img_f in img_files:
            if img_f.name in tex_dict:
                tex_f = tex_dict[img_f.name]
                filename = img_f.name
                self.samples.append((str(img_f), tex_f, filename))
            else:
                logging.warning(
                    f"No matching texture found for {img_f.name} in subfolders of {tex_dir}"
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_file, texture_file, filename = self.samples[idx]
        image = self.load_image(image_file, mode="RGB")
        texture = self.load_image(texture_file, mode="L")

        if self.img_size is not None:
            image = image.resize((self.img_size, self.img_size), Image.BILINEAR)
            texture = texture.resize((self.img_size, self.img_size), Image.BILINEAR)

        np_img = np.array(image, dtype=np.float32)
        np_tex = np.array(texture, dtype=np.float32)
        np_img = np_img.transpose((2, 0, 1))
        np_tex = np_tex[np.newaxis, ...]

        if np_img.max() > 1.0:
            np_img = np_img / 255.0
            np_img = np_img * 2.0 - 1.0
        if np_tex.max() > 1.0:
            np_tex = np_tex / 255.0
            np_tex = np_tex * 2.0 - 1.0

        combined = np.concatenate([np_img, np_tex], axis=0)
        return torch.tensor(combined, dtype=torch.float32), filename

    @staticmethod
    def load_image(path, mode="RGB"):
        return Image.open(path).convert(mode)


def create_model(args):
    net_name = args.network.lower()
    if net_name == "unet":
        model = UNet(
            n_channels=args.num_channels, n_classes=args.num_classes, bilinear=True
        )
    elif net_name == "swinunetr":
        model = SwinUNETR(
            in_channels=args.num_channels,
            out_channels=args.num_classes,
        )
    else:
        raise ValueError(f"Unsupported model: {args.network}")
    return model


def load_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict)
    return model


def mask_to_image(mask: np.ndarray):
    mask = (mask > 0).astype(np.uint8) * 255
    return Image.fromarray(mask)


def run_inference(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    dataset = WrinkleTestDataset(
        image_path=args.image_path,
        texture_path=args.texture_path,
        img_size=args.img_size,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = create_model(args)
    model.to(device)
    load_checkpoint(model, args.checkpoint, device)
    model.eval()
    logging.info(f"Loaded checkpoint from {args.checkpoint}")

    os.makedirs(args.output_dir, exist_ok=True)

    with torch.inference_mode():
        for inputs, filename in tqdm(loader, desc="Inference", unit="batch"):
            if isinstance(filename, (list, tuple)):
                filename = filename[0]
            inputs = inputs.to(device)
            preds = model(inputs)
            pred_mask = preds.argmax(dim=1)

            pred_mask_np = pred_mask[0].cpu().numpy()
            result_img = mask_to_image(pred_mask_np)

            base_stem, ext = os.path.splitext(filename)
            out_path = os.path.join(args.output_dir, f"{base_stem}_mask{ext}")
            result_img.save(out_path)


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    run_inference(args)


if __name__ == "__main__":
    main()
