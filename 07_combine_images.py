# -*- coding: utf-8 -*-

from PIL import Image
from pathlib import Path
import argparse


def combine_vertical(image1, image2, output, gap=40, margin=30, background="white"):
    img1 = Image.open(image1).convert("RGB")
    img2 = Image.open(image2).convert("RGB")

    target_width = max(img1.width, img2.width)

    def resize_to_width(img, width):
        if img.width == width:
            return img
        new_height = int(img.height * width / img.width)
        return img.resize((width, new_height), Image.LANCZOS)

    img1 = resize_to_width(img1, target_width)
    img2 = resize_to_width(img2, target_width)

    canvas_width = target_width + margin * 2
    canvas_height = img1.height + img2.height + margin * 2 + gap

    canvas = Image.new("RGB", (canvas_width, canvas_height), background)
    canvas.paste(img1, (margin, margin))
    canvas.paste(img2, (margin, margin + img1.height + gap))

    canvas.save(output, quality=95)
    print(f"已生成：{output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="把两张图片上下拼接成一张图")
    parser.add_argument("image1", help="第一张图片路径")
    parser.add_argument("image2", help="第二张图片路径")
    parser.add_argument("-o", "--output", default="合并图片.png", help="输出图片路径")
    parser.add_argument("--gap", type=int, default=40, help="两张图之间的空白")
    parser.add_argument("--margin", type=int, default=30, help="边距")

    args = parser.parse_args()

    combine_vertical(
        image1=args.image1,
        image2=args.image2,
        output=args.output,
        gap=args.gap,
        margin=args.margin
    )
