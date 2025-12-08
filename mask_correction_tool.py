# mask_correction_tool.py
"""Interactive GUI tool for correcting segmentation masks."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
from pathlib import Path
from PIL import Image
import cv2


class MaskEditor:
    """Interactive mask editor with brush tool."""

    def __init__(self, base_img: np.ndarray, clothed_img: np.ndarray, mask: np.ndarray):
        """Initialize editor.

        Args:
            base_img: Base character image (512, 512, 3) RGB uint8
            clothed_img: Clothed character image (512, 512, 3) RGB uint8
            mask: Binary mask (512, 512) uint8 where 1=clothing
        """
        self.base_img = base_img
        self.clothed_img = clothed_img
        self.mask = mask.copy()
        self.original_mask = mask.copy()

        self.brush_size = 5
        self.paint_mode = 1  # 1=add clothing, 0=remove clothing
        self.is_painting = False

        # Setup figure
        self.fig, self.axes = plt.subplots(1, 3, figsize=(15, 5))
        self.fig.suptitle('Mask Correction Tool - Left Click: Paint | Right Click: Erase | Scroll: Brush Size')

        # Display images
        self.axes[0].imshow(self.clothed_img)
        self.axes[0].set_title('Clothed Frame')
        self.axes[0].axis('off')

        self.axes[1].imshow(self.base_img)
        self.axes[1].set_title('Base Frame')
        self.axes[1].axis('off')

        self.mask_display = self.axes[2].imshow(self.get_overlay(), alpha=1.0)
        self.axes[2].set_title('Mask Overlay (Red=Clothing)')
        self.axes[2].axis('off')

        # Connect events
        self.fig.canvas.mpl_connect('button_press_event', self.on_mouse_down)
        self.fig.canvas.mpl_connect('button_release_event', self.on_mouse_up)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)

        # Add buttons
        self.add_buttons()

    def get_overlay(self) -> np.ndarray:
        """Create red overlay showing mask on clothed image."""
        overlay = self.clothed_img.copy().astype(np.float32) / 255.0

        # Red tint for masked regions
        overlay[self.mask == 1] = [1.0, 0.0, 0.0]

        return overlay

    def paint_at(self, x: int, y: int, value: int, brush_size: int):
        """Paint circular brush at position.

        Args:
            x: X coordinate
            y: Y coordinate
            value: 0 or 1 (remove or add clothing)
            brush_size: Brush radius in pixels
        """
        # Create circular brush
        for dy in range(-brush_size, brush_size + 1):
            for dx in range(-brush_size, brush_size + 1):
                if dx*dx + dy*dy <= brush_size*brush_size:
                    py, px = y + dy, x + dx
                    if 0 <= py < 512 and 0 <= px < 512:
                        self.mask[py, px] = value

    def on_mouse_down(self, event):
        """Handle mouse button press."""
        if event.inaxes != self.axes[2]:
            return

        self.is_painting = True

        # Left click = add clothing (1), Right click = remove (0)
        self.paint_mode = 1 if event.button == 1 else 0

        x, y = int(event.xdata), int(event.ydata)
        self.paint_at(x, y, self.paint_mode, self.brush_size)
        self.update_display()

    def on_mouse_up(self, event):
        """Handle mouse button release."""
        self.is_painting = False

    def on_mouse_move(self, event):
        """Handle mouse movement."""
        if not self.is_painting or event.inaxes != self.axes[2]:
            return

        x, y = int(event.xdata), int(event.ydata)
        self.paint_at(x, y, self.paint_mode, self.brush_size)
        self.update_display()

    def on_scroll(self, event):
        """Handle scroll to change brush size."""
        if event.button == 'up':
            self.brush_size = min(50, self.brush_size + 2)
        else:
            self.brush_size = max(1, self.brush_size - 2)

        self.fig.suptitle(f'Brush Size: {self.brush_size} | Left Click: Paint | Right Click: Erase')
        self.fig.canvas.draw_idle()

    def on_key_press(self, event):
        """Handle keyboard shortcuts."""
        if event.key == 'r':  # Reset to original
            self.mask = self.original_mask.copy()
            self.update_display()
        elif event.key == 'c':  # Clear all
            self.mask.fill(0)
            self.update_display()

    def update_display(self):
        """Refresh mask overlay display."""
        self.mask_display.set_data(self.get_overlay())
        self.fig.canvas.draw_idle()

    def add_buttons(self):
        """Add save/cancel buttons."""
        # Save button
        save_ax = plt.axes([0.7, 0.01, 0.1, 0.04])
        save_btn = Button(save_ax, 'Save')
        save_btn.on_clicked(lambda e: self.save_and_close())

        # Cancel button
        cancel_ax = plt.axes([0.82, 0.01, 0.1, 0.04])
        cancel_btn = Button(cancel_ax, 'Cancel')
        cancel_btn.on_clicked(lambda e: plt.close())

    def save_and_close(self):
        """Mark as saved and close."""
        self.saved = True
        plt.close()

    def show(self) -> np.ndarray:
        """Show editor and return corrected mask."""
        self.saved = False
        plt.show()
        return self.mask if self.saved else None


def correct_mask_interactive(frame_num: int, frames_dir: Path, masks_dir: Path, output_dir: Path):
    """Interactively correct a single mask.

    Args:
        frame_num: Frame number (0-24)
        frames_dir: Directory containing frame images
        masks_dir: Directory containing initial masks
        output_dir: Directory to save corrected masks
    """
    # Load images
    base_path = frames_dir / f"base_frame_{frame_num:02d}.png"
    clothed_path = frames_dir / f"clothed_frame_{frame_num:02d}.png"
    mask_path = masks_dir / f"mask_{frame_num:02d}.png"

    base = np.array(Image.open(base_path).convert('RGB'))
    clothed = np.array(Image.open(clothed_path).convert('RGB'))
    mask = np.array(Image.open(mask_path).convert('L')) // 255

    # Edit mask
    editor = MaskEditor(base, clothed, mask)
    corrected_mask = editor.show()

    if corrected_mask is not None:
        # Save corrected mask
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"mask_{frame_num:02d}.png"

        mask_img = Image.fromarray(corrected_mask * 255)
        mask_img.save(output_path)

        # Statistics
        clothing_pixels = np.sum(corrected_mask == 1)
        percent = 100 * clothing_pixels / (512 * 512)
        print(f"Frame {frame_num:02d}: ✓ Saved ({clothing_pixels} pixels, {percent:.1f}%)")
        return True
    else:
        print(f"Frame {frame_num:02d}: Cancelled")
        return False


def correct_all_masks(frames_dir: Path, initial_masks_dir: Path, corrected_masks_dir: Path):
    """Correct all 25 masks interactively."""
    print("Mask Correction Tool")
    print("=" * 70)
    print("Controls:")
    print("  - Left Click: Add clothing pixels (paint red)")
    print("  - Right Click: Remove clothing pixels (erase red)")
    print("  - Scroll: Adjust brush size")
    print("  - 'R' key: Reset to initial mask")
    print("  - 'C' key: Clear all (start from scratch)")
    print("=" * 70)
    print()

    for frame_num in range(25):
        output_path = corrected_masks_dir / f"mask_{frame_num:02d}.png"

        if output_path.exists():
            response = input(f"Frame {frame_num:02d} already corrected. Re-edit? (y/N): ")
            if response.lower() != 'y':
                print(f"Frame {frame_num:02d}: Skipping")
                continue

        print(f"\nEditing Frame {frame_num:02d}/24...")
        success = correct_mask_interactive(frame_num, frames_dir, initial_masks_dir, corrected_masks_dir)

        if not success:
            print("Correction cancelled. Exiting.")
            break

    print()
    print("=" * 70)
    print(f"✓ Corrected masks saved to {corrected_masks_dir}/")


if __name__ == "__main__":
    frames_dir = Path("training_data/frames")
    initial_masks_dir = Path("training_data/masks_initial")
    corrected_masks_dir = Path("training_data/masks_corrected")

    correct_all_masks(frames_dir, initial_masks_dir, corrected_masks_dir)
