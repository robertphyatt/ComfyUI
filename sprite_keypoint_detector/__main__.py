"""Command-line interface for sprite keypoint detector."""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='Sprite Keypoint Detector')
    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Annotate
    ann = subparsers.add_parser('annotate', help='Annotate keypoints')
    ann.add_argument('image_dir', type=Path)
    ann.add_argument('--output', '-o', type=Path, default=None)
    ann.add_argument('--seed', type=Path, default=None, help='Seed annotations JSON')
    ann.add_argument('--pattern', '-p', type=str, default='*.png')

    # Train
    tr = subparsers.add_parser('train', help='Train detector')
    tr.add_argument('annotations', type=Path)
    tr.add_argument('image_dir', type=Path)
    tr.add_argument('--output', '-o', type=Path, default=Path('models/sprite_keypoint'))
    tr.add_argument('--epochs', '-e', type=int, default=100)
    tr.add_argument('--batch-size', '-b', type=int, default=8)
    tr.add_argument('--lr', type=float, default=1e-3)
    tr.add_argument('--device', type=str, default=None)

    # Predict
    pr = subparsers.add_parser('predict', help='Detect keypoints')
    pr.add_argument('model', type=Path)
    pr.add_argument('images', type=Path, nargs='+')
    pr.add_argument('--output-dir', '-o', type=Path, default=None)
    pr.add_argument('--skeleton-only', action='store_true')

    args = parser.parse_args()

    if args.command == 'annotate':
        from .annotator import annotate_directory
        output = args.output or (args.image_dir / 'annotations.json')
        annotate_directory(args.image_dir, output, args.pattern, seed_json=args.seed)

    elif args.command == 'train':
        from .train import train
        train(args.annotations, args.image_dir, args.output,
              epochs=args.epochs, batch_size=args.batch_size,
              learning_rate=args.lr, device=args.device)

    elif args.command == 'predict':
        from .inference import SpriteKeypointPredictor, draw_skeleton, render_skeleton_only
        from PIL import Image

        predictor = SpriteKeypointPredictor(args.model)
        output_dir = args.output_dir or Path('.')
        output_dir.mkdir(parents=True, exist_ok=True)

        for img_path in args.images:
            keypoints = predictor.predict(img_path)
            print(f"\n{img_path.name}:")
            for name, (x, y) in keypoints.items():
                print(f"  {name}: ({x}, {y})")

            if args.skeleton_only:
                result = render_skeleton_only(keypoints)
            else:
                image = Image.open(img_path).convert('RGB')
                result = draw_skeleton(image, keypoints)

            output_path = output_dir / f"{img_path.stem}_skeleton.png"
            Image.fromarray(result).save(output_path)
            print(f"  Saved: {output_path}")
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
