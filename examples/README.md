# Inference Pipeline Example 

This notebook demonstrates a **complete inference pipeline** that processes a single input image through a series of pre-processing and deep learning model steps, in alignment with the method described in the paper:

> **Facial Wrinkle Segmentation for Cosmetic Dermatology: Pretraining with Texture Map-Based Weak Supervision**  
> *Yuyang Zhao, Tongzhou Wang, Antonio Torralba*  
> [arXiv:2408.10060](https://arxiv.org/abs/2408.10060)

---

## Pipeline Summary

1. **Image Upload & Preprocessing**
   - Accept user image upload (`.png`).
   - Pad to square, resize to 1024×1024 resolution.
   - Store the raw and resized images.

2. **Face Parsing**
   - Use a pre-trained [BiSeNet](https://github.com/zllrunning/face-parsing.PyTorch) model for semantic segmentation (face regions).
   - Save the resulting segmentation as `.npy`.

3. **Mask Generation**
   - Extract key face regions into a binary mask.
   - Apply this mask to isolate face pixels from RGB image and grayscale texture map.

4. **Texture Map Creation**
   - Convert image to grayscale and apply Gaussian blur.
   - Generate a basic texture representation based on local contrast.
   - Mask the texture map using the face mask.

5. **Model Inference**
   - Run the main inference script (`inference.py`) with the masked RGB + texture map as input.
   - Model outputs prediction masks stored in `results` folder.

6. **Visualization**
   - Show original image, masked image, texture map, predicted mask, and overlay visualization.
   - Can be shown for a single image or all processed images.

---

## Notes

- This pipeline is designed to run on **Google Colab**, and **does not require a GPU**.  
- Some steps assume pre-initialized global paths (e.g. `paths["input"]`, `paths["masked"]`), and that checkpoints are already downloaded (e.g. `FACEPARSING_CKPT`, `CKPT`).

---

## Author

**Contributor**: Hoai Linh DAO  
**GitHub**: [https://github.com/dhlinhdn00](https://github.com/dhlinhdn00)  
**Email**: daohoailinhdn00@gmail.com
