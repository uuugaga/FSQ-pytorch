#!/bin/bash

# Example usage: bash ./eval.sh 0 /path/to/real/images /path/to/fake/images

# Get the input of device number
if [ -z "$1" ]; then
    echo "Please provide the device number as an argument."
    exit 1
fi
DEVICE=$1

# Get real images path and fake images path from the command line
if [ -z "$2" ]; then
    echo "Please provide the path to the real images."
    exit 1
fi
REAL_IMAGES_PATH=$2

if [ -z "$3" ]; then
    echo "Please provide the path to the fake images."
    exit 1
fi
FAKE_IMAGES_PATH=$3

# test FID and get the last line of the output
FID_OUTPUT=$(python eval_fid.py --real $REAL_IMAGES_PATH --gen $FAKE_IMAGES_PATH --device cuda:$DEVICE)

# test lpips
LPIPS_OUTPUT=$(python eval_lpips.py --real $REAL_IMAGES_PATH --gen $FAKE_IMAGES_PATH --device cuda:$DEVICE)

# test ssim
SSIM_OUTPUT=$(python eval_ssim.py --real $REAL_IMAGES_PATH --gen $FAKE_IMAGES_PATH --device cuda:$DEVICE)

# test psnr
PSNR_OUTPUT=$(python eval_psnr.py --real $REAL_IMAGES_PATH --gen $FAKE_IMAGES_PATH --device cuda:$DEVICE)

# Output the results ({Metric}: {Value}) in a table format, I only want the {Value} part
echo "Evaluation Results:"
echo "-------------------------------------"
echo "| Metric | Value                     |"
echo "-------------------------------------"
echo "| FID    | $(echo "$FID_OUTPUT" | tail -n 1 | cut -d':' -f2 | xargs) |"
echo "| LPIPS  | $(echo "$LPIPS_OUTPUT" | tail -n 1 | cut -d':' -f2 | xargs) |"
echo "| SSIM   | $(echo "$SSIM_OUTPUT" | tail -n 1 | cut -d':' -f2 | xargs) |"
echo "| PSNR   | $(echo "$PSNR_OUTPUT" | tail -n 1 | cut -d':' -f2 | xargs) |"
echo "-------------------------------------"
echo "Evaluation completed."