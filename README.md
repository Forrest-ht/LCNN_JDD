# LCNN_JDD
## Code for Lightweight Deep Residue Learning for Joint Color Image Demosaicking and Denoising (ICPR2018 Oral)


### train code

  python main_ICPR.py --is_train 1 --train_dataset './data/WED/'


### test code

  python main_ICPR.py --is_train 0 --test_dataset './data/Kodak/'


### avarage results for Kodak and McMaster dataset (PSNR and SSIM)

        | dataset | LCNN-DD       | LCNN-DD+      |
        | :-----  | ----:         | :----:        |  
        | Kodak   | 42.55dB/99.57 | 45.66dB/99.82 |
        | McMaster| 39.25dB/99.01 | 42.03dB/99.68 |
  