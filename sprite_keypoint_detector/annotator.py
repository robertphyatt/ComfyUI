"""Interactive keypoint annotation tool for sprite frames."""

import json
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from pathlib import Path
from PIL import Image
import numpy as np
from typing import Optional, Dict, List, Tuple

from .keypoints import KEYPOINT_NAMES, NUM_KEYPOINTS, SKELETON_CONNECTIONS, SKELETON_COLORS


class KeypointAnnotator:
    """Interactive GUI for annotating keypoints on sprite images."""

    def __init__(self, image_path: Path, existing_keypoints: Optional[Dict] = None):
        """Initialize annotator.

        Args:
            image_path: Path to sprite image
            existing_keypoints: Optional dict of existing keypoint annotations
        """
        self.image_path = Path(image_path)
        self.image = np.array(Image.open(image_path).convert('RGBA'))

        # Initialize keypoints: None means not yet annotated
        self.keypoints: List[Optional[Tuple[int, int]]] = [None] * NUM_KEYPOINTS

        # Load existing keypoints if provided
        if existing_keypoints:
            for i, name in enumerate(KEYPOINT_NAMES):
                if name in existing_keypoints:
                    self.keypoints[i] = tuple(existing_keypoints[name])

        self.current_keypoint_idx = 0
        self.point_artists = []
        self.line_artists = []
        self.saved = False

        self._setup_gui()

    def _setup_gui(self):
        """Set up the matplotlib GUI."""
        self.fig, self.ax = plt.subplots(1, 1, figsize=(10, 10))
        self.fig.canvas.manager.set_window_title(f'Annotate: {self.image_path.name}')

        # Display image
        self.ax.imshow(self.image)
        self.ax.set_title(self._get_title())
        self.ax.axis('off')

        # Draw existing keypoints
        self._draw_skeleton()

        # Connect events
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)

        # Add buttons
        ax_save = plt.axes([0.7, 0.02, 0.1, 0.04])
        ax_skip = plt.axes([0.81, 0.02, 0.1, 0.04])
        ax_prev = plt.axes([0.1, 0.02, 0.1, 0.04])
        ax_next = plt.axes([0.21, 0.02, 0.1, 0.04])
        ax_clear = plt.axes([0.4, 0.02, 0.15, 0.04])

        self.btn_save = Button(ax_save, 'Save')
        self.btn_skip = Button(ax_skip, 'Skip')
        self.btn_prev = Button(ax_prev, '< Prev')
        self.btn_next = Button(ax_next, 'Next >')
        self.btn_clear = Button(ax_clear, 'Clear Current')

        self.btn_save.on_clicked(self._on_save)
        self.btn_skip.on_clicked(self._on_skip)
        self.btn_prev.on_clicked(self._on_prev)
        self.btn_next.on_clicked(self._on_next)
        self.btn_clear.on_clicked(self._on_clear)

        # Instructions
        self.fig.text(0.5, 0.97,
            'Click to place keypoint | Arrow keys or buttons to navigate | S to save | Q to quit',
            ha='center', va='top', fontsize=9, color='gray')

    def _get_title(self) -> str:
        """Get title showing current keypoint."""
        name = KEYPOINT_NAMES[self.current_keypoint_idx]
        status = "SET" if self.keypoints[self.current_keypoint_idx] else "NOT SET"
        annotated = sum(1 for kp in self.keypoints if kp is not None)
        return f'[{self.current_keypoint_idx + 1}/{NUM_KEYPOINTS}] {name} ({status}) - {annotated}/{NUM_KEYPOINTS} annotated'

    def _draw_skeleton(self):
        """Draw all keypoints and skeleton connections."""
        # Clear existing artists
        for artist in self.point_artists + self.line_artists:
            artist.remove()
        self.point_artists = []
        self.line_artists = []

        # Draw skeleton lines
        for (i, j), color in zip(SKELETON_CONNECTIONS, SKELETON_COLORS):
            if self.keypoints[i] is not None and self.keypoints[j] is not None:
                x = [self.keypoints[i][0], self.keypoints[j][0]]
                y = [self.keypoints[i][1], self.keypoints[j][1]]
                # Normalize RGB
                rgb = (color[0]/255, color[1]/255, color[2]/255)
                line, = self.ax.plot(x, y, '-', color=rgb, linewidth=2, alpha=0.8)
                self.line_artists.append(line)

        # Draw keypoints
        for i, kp in enumerate(self.keypoints):
            if kp is not None:
                color = 'lime' if i == self.current_keypoint_idx else 'cyan'
                size = 120 if i == self.current_keypoint_idx else 60
                point = self.ax.scatter(kp[0], kp[1], c=color, s=size,
                                        marker='o', edgecolors='white', linewidths=1, zorder=10)
                self.point_artists.append(point)

                # Add label for current keypoint
                if i == self.current_keypoint_idx:
                    text = self.ax.text(kp[0] + 10, kp[1] - 10, KEYPOINT_NAMES[i],
                                       fontsize=8, color='white',
                                       bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
                    self.point_artists.append(text)

        self.fig.canvas.draw_idle()

    def _on_click(self, event):
        """Handle mouse click to place keypoint."""
        if event.inaxes != self.ax:
            return
        if event.button != 1:  # Left click only
            return

        x, y = int(round(event.xdata)), int(round(event.ydata))
        self.keypoints[self.current_keypoint_idx] = (x, y)

        # Auto-advance to next unset keypoint
        self._advance_to_next_unset()

        self.ax.set_title(self._get_title())
        self._draw_skeleton()

    def _advance_to_next_unset(self):
        """Advance to the next unset keypoint, or stay if all are set."""
        start = self.current_keypoint_idx
        for _ in range(NUM_KEYPOINTS):
            self.current_keypoint_idx = (self.current_keypoint_idx + 1) % NUM_KEYPOINTS
            if self.keypoints[self.current_keypoint_idx] is None:
                return
        # All set, stay at current + 1 (or wrap)
        self.current_keypoint_idx = (start + 1) % NUM_KEYPOINTS

    def _on_key(self, event):
        """Handle keyboard input."""
        if event.key == 'right' or event.key == 'down':
            self.current_keypoint_idx = (self.current_keypoint_idx + 1) % NUM_KEYPOINTS
        elif event.key == 'left' or event.key == 'up':
            self.current_keypoint_idx = (self.current_keypoint_idx - 1) % NUM_KEYPOINTS
        elif event.key == 's':
            self._on_save(None)
            return
        elif event.key == 'q':
            plt.close(self.fig)
            return
        elif event.key == 'c':
            self._on_clear(None)
            return

        self.ax.set_title(self._get_title())
        self._draw_skeleton()

    def _on_save(self, event):
        """Save and close."""
        self.saved = True
        plt.close(self.fig)

    def _on_skip(self, event):
        """Skip without saving."""
        self.saved = False
        plt.close(self.fig)

    def _on_prev(self, event):
        """Go to previous keypoint."""
        self.current_keypoint_idx = (self.current_keypoint_idx - 1) % NUM_KEYPOINTS
        self.ax.set_title(self._get_title())
        self._draw_skeleton()

    def _on_next(self, event):
        """Go to next keypoint."""
        self.current_keypoint_idx = (self.current_keypoint_idx + 1) % NUM_KEYPOINTS
        self.ax.set_title(self._get_title())
        self._draw_skeleton()

    def _on_clear(self, event):
        """Clear current keypoint."""
        self.keypoints[self.current_keypoint_idx] = None
        self.ax.set_title(self._get_title())
        self._draw_skeleton()

    def run(self) -> Optional[Dict[str, List[int]]]:
        """Run the annotator and return keypoints if saved.

        Returns:
            Dict mapping keypoint names to [x, y] coordinates, or None if skipped
        """
        plt.show()

        if not self.saved:
            return None

        # Convert to dict format
        result = {}
        for i, name in enumerate(KEYPOINT_NAMES):
            if self.keypoints[i] is not None:
                result[name] = list(self.keypoints[i])

        return result


def load_seed_annotations(seed_json: Path) -> Dict:
    """Load seed annotations and convert to annotation tool format.

    Args:
        seed_json: Path to seed_annotations.json (AI-generated initial keypoints)

    Returns:
        Dict in annotation tool format
    """
    with open(seed_json) as f:
        seed_data = json.load(f)

    # Convert from seed format to annotation format
    annotations = {}
    for img_name, data in seed_data.get("annotations", {}).items():
        keypoints = data.get("keypoints", {})
        annotations[img_name] = {
            "image": img_name,
            "keypoints": keypoints
        }

    return annotations


def annotate_directory(
    image_dir: Path,
    output_json: Path,
    pattern: str = "*.png",
    seed_json: Optional[Path] = None
) -> None:
    """Annotate all images in a directory.

    Args:
        image_dir: Directory containing images
        output_json: Path to save/load annotations JSON
        pattern: Glob pattern for image files
        seed_json: Optional path to seed annotations (AI-generated initial keypoints)
    """
    image_dir = Path(image_dir)
    output_json = Path(output_json)

    # Load existing annotations
    annotations = {}
    if output_json.exists():
        with open(output_json) as f:
            annotations = json.load(f)
        print(f"Loaded {len(annotations)} existing annotations")
    elif seed_json and Path(seed_json).exists():
        # Load from seed annotations if no existing annotations
        annotations = load_seed_annotations(seed_json)
        print(f"Loaded {len(annotations)} seed annotations for review")

    # Get all images
    images = sorted(image_dir.glob(pattern))
    print(f"Found {len(images)} images")

    # Annotate each
    for i, img_path in enumerate(images):
        img_name = img_path.name
        existing = annotations.get(img_name, {}).get("keypoints")

        status = "DONE" if existing and len(existing) == NUM_KEYPOINTS else "TODO"
        print(f"\n[{i+1}/{len(images)}] {img_name} ({status})")

        if status == "DONE":
            resp = input("  Already annotated. Re-annotate? (y/N): ").strip().lower()
            if resp != 'y':
                continue

        annotator = KeypointAnnotator(img_path, existing)
        result = annotator.run()

        if result:
            annotations[img_name] = {
                "image": img_name,
                "keypoints": result
            }
            # Save after each annotation
            with open(output_json, 'w') as f:
                json.dump(annotations, f, indent=2)
            print(f"  Saved {len(result)} keypoints")
        else:
            print("  Skipped")

    print(f"\nDone! Total annotations: {len(annotations)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Interactive keypoint annotation tool for sprite frames"
    )
    parser.add_argument(
        "image_dir",
        type=Path,
        help="Directory containing sprite images"
    )
    parser.add_argument(
        "output_json",
        type=Path,
        nargs='?',
        help="Path to save/load annotations JSON (default: <image_dir>/annotations.json)"
    )
    parser.add_argument(
        "-p", "--pattern",
        default="*.png",
        help="Glob pattern for image files (default: *.png)"
    )
    parser.add_argument(
        "--seed",
        type=Path,
        help="Path to seed annotations JSON (AI-generated initial keypoints)"
    )

    args = parser.parse_args()

    # Default output_json if not provided
    output_json = args.output_json if args.output_json else args.image_dir / "annotations.json"

    annotate_directory(
        args.image_dir,
        output_json,
        pattern=args.pattern,
        seed_json=args.seed
    )
