# mask_correction_tool.py
"""Interactive GUI tool for correcting segmentation masks."""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
from pathlib import Path
from PIL import Image
import cv2

# Alpha threshold for transparent background detection
ALPHA_THRESHOLD = 128


def remove_transparent_background(mask: np.ndarray, base_img_rgba: np.ndarray) -> np.ndarray:
    """Remove any clothing labels from transparent background pixels.

    Args:
        mask: Binary mask (H, W) uint8, where 1=clothing
        base_img_rgba: Base image with alpha channel (H, W, 4) uint8

    Returns:
        Cleaned mask with transparent pixels zeroed out
    """
    if base_img_rgba.shape[2] != 4:
        # No alpha channel, return mask unchanged
        return mask

    # Get alpha channel
    alpha = base_img_rgba[:, :, 3]

    # Zero out mask where background is transparent
    cleaned_mask = mask.copy()
    cleaned_mask[alpha < ALPHA_THRESHOLD] = 0

    return cleaned_mask


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
        self.zoom_level = 1.0

        # Undo/redo stacks
        self.undo_stack = []
        self.redo_stack = []
        self.max_undo_levels = 50

        # Setup figure
        self.fig, self.axes = plt.subplots(1, 3, figsize=(15, 5))
        self.update_title()

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

    def save_state(self):
        """Save current mask state to undo stack."""
        # Save a copy of current mask
        self.undo_stack.append(self.mask.copy())

        # Limit undo stack size
        if len(self.undo_stack) > self.max_undo_levels:
            self.undo_stack.pop(0)

        # Clear redo stack when new action is performed
        self.redo_stack.clear()

        self.update_title()

    def undo(self):
        """Undo last change."""
        if not self.undo_stack:
            return

        # Save current state to redo stack
        self.redo_stack.append(self.mask.copy())

        # Restore previous state
        self.mask = self.undo_stack.pop()
        self.update_display()
        self.update_title()

    def redo(self):
        """Redo last undone change."""
        if not self.redo_stack:
            return

        # Save current state to undo stack
        self.undo_stack.append(self.mask.copy())

        # Restore redone state
        self.mask = self.redo_stack.pop()
        self.update_display()
        self.update_title()

    def on_mouse_down(self, event):
        """Handle mouse button press."""
        if event.inaxes != self.axes[2]:
            return

        # Save state before starting to paint
        self.save_state()

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
        """Handle scroll to change brush size or zoom (with Ctrl)."""
        # Ctrl+Scroll = zoom on mask overlay
        if event.key == 'control' and event.inaxes == self.axes[2]:
            self.zoom_on_mask(event)
        # Regular scroll = brush size
        else:
            if event.button == 'up':
                self.brush_size = min(50, self.brush_size + 2)
            else:
                self.brush_size = max(1, self.brush_size - 2)
            self.update_title()

        self.fig.canvas.draw_idle()

    def on_key_press(self, event):
        """Handle keyboard shortcuts."""
        key_str = str(event.key).lower()

        # Undo: Cmd+Z (Mac) or Ctrl+Z (Windows/Linux)
        # On Mac: 'cmd+z' or 'super+z'
        # On Windows/Linux: 'ctrl+z'
        if 'z' in key_str and ('cmd' in key_str or 'ctrl' in key_str or 'super' in key_str):
            if 'shift' in key_str:
                self.redo()  # Cmd+Shift+Z = Redo
            else:
                self.undo()  # Cmd+Z = Undo
        elif event.key == 'r':  # Reset to original
            self.save_state()
            self.mask = self.original_mask.copy()
            self.update_display()
        elif event.key == 'c':  # Clear all
            self.save_state()
            self.mask.fill(0)
            self.update_display()
        elif event.key == '+' or event.key == '=':  # Zoom in
            self.zoom_level = min(8.0, self.zoom_level * 1.5)
            self.apply_zoom()
        elif event.key == '-':  # Zoom out
            self.zoom_level = max(1.0, self.zoom_level / 1.5)
            self.apply_zoom()
        elif event.key == '0':  # Reset zoom
            self.zoom_level = 1.0
            self.apply_zoom()

    def update_display(self):
        """Refresh mask overlay display."""
        self.mask_display.set_data(self.get_overlay())
        self.fig.canvas.draw_idle()

    def update_title(self):
        """Update figure title with current tool state."""
        # Build title components
        parts = []

        # Zoom level
        if self.zoom_level > 1.0:
            parts.append(f"Zoom: {self.zoom_level:.1f}x")

        # Undo/Redo status
        undo_str = f"Undo: {len(self.undo_stack)}" if self.undo_stack else "Undo: -"
        redo_str = f"Redo: {len(self.redo_stack)}" if self.redo_stack else "Redo: -"
        parts.append(f"{undo_str}/{redo_str}")

        # Controls
        parts.append(f"Brush: {self.brush_size}")
        parts.append("Cmd+Z: Undo")

        self.fig.suptitle(" | ".join(parts))

    def zoom_on_mask(self, event):
        """Zoom in/out on mask overlay centered on mouse position."""
        if event.xdata is None or event.ydata is None:
            return

        # Get current axis limits
        xlim = self.axes[2].get_xlim()
        ylim = self.axes[2].get_ylim()

        # Mouse position in data coordinates
        xdata, ydata = event.xdata, event.ydata

        # Zoom factor
        if event.button == 'up':
            scale_factor = 0.75  # Zoom in
            self.zoom_level = min(8.0, self.zoom_level / 0.75)
        else:
            scale_factor = 1.25  # Zoom out
            self.zoom_level = max(1.0, self.zoom_level * 0.75)

        # Calculate new limits centered on mouse
        new_width = (xlim[1] - xlim[0]) * scale_factor
        new_height = (ylim[1] - ylim[0]) * scale_factor

        relx = (xlim[1] - xdata) / (xlim[1] - xlim[0])
        rely = (ylim[1] - ydata) / (ylim[1] - ylim[0])

        new_xlim = [xdata - new_width * (1 - relx), xdata + new_width * relx]
        new_ylim = [ydata - new_height * (1 - rely), ydata + new_height * rely]

        # Apply limits
        self.axes[2].set_xlim(new_xlim)
        self.axes[2].set_ylim(new_ylim)
        self.update_title()

    def apply_zoom(self):
        """Apply current zoom level to mask overlay."""
        # Center on current view
        xlim = self.axes[2].get_xlim()
        ylim = self.axes[2].get_ylim()

        center_x = (xlim[0] + xlim[1]) / 2
        center_y = (ylim[0] + ylim[1]) / 2

        # Calculate new extent based on zoom level
        base_width = 512
        base_height = 512

        new_width = base_width / (2 * self.zoom_level)
        new_height = base_height / (2 * self.zoom_level)

        # Set new limits
        self.axes[2].set_xlim([center_x - new_width, center_x + new_width])
        self.axes[2].set_ylim([center_y + new_height, center_y - new_height])

        self.update_title()
        self.fig.canvas.draw_idle()

    def add_buttons(self):
        """Add save/cancel buttons."""
        # Save button (store as instance var to prevent garbage collection)
        save_ax = plt.axes([0.7, 0.01, 0.1, 0.04])
        self.save_btn = Button(save_ax, 'Save')
        self.save_btn.on_clicked(lambda e: self.save_and_close())

        # Cancel button (store as instance var to prevent garbage collection)
        cancel_ax = plt.axes([0.82, 0.01, 0.1, 0.04])
        self.cancel_btn = Button(cancel_ax, 'Cancel')
        self.cancel_btn.on_clicked(lambda e: plt.close())

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

    base_rgba = np.array(Image.open(base_path).convert('RGBA'))
    base_rgb = base_rgba[:, :, :3]  # Extract RGB for display
    clothed = np.array(Image.open(clothed_path).convert('RGB'))
    mask = np.array(Image.open(mask_path).convert('L')) // 255

    # Edit mask
    editor = MaskEditor(base_rgb, clothed, mask)
    corrected_mask = editor.show()

    if corrected_mask is not None:
        # Remove any clothing labels from transparent background
        corrected_mask = remove_transparent_background(corrected_mask, base_rgba)

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
    print("  - Ctrl+Scroll: Zoom in/out (centered on mouse)")
    print("  - Cmd+Z / Ctrl+Z: Undo last change")
    print("  - Cmd+Shift+Z / Ctrl+Shift+Z: Redo")
    print("  - +/- keys: Zoom in/out")
    print("  - 0 key: Reset zoom to 1x")
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
    parser = argparse.ArgumentParser(
        description="Interactive mask correction tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default (training_data/)
  python mask_correction_tool.py

  # Custom directory
  python mask_correction_tool.py training_data_walk_south

  # Fully custom paths
  python mask_correction_tool.py --frames my/frames --initial my/masks_initial --output my/masks_corrected
"""
    )
    parser.add_argument("data_dir", type=Path, nargs="?", default=None,
                       help="Base data directory (contains frames/, masks_initial/)")
    parser.add_argument("--animation", type=str, default=None,
                       help="Animation name (e.g., 'walk_south'). Auto-configures paths to "
                            "training_data/animations/<name>/. Output writes to same masks/ dir.")
    parser.add_argument("--frames", type=Path, default=None,
                       help="Frames directory (default: <data_dir>/frames)")
    parser.add_argument("--initial", type=Path, default=None,
                       help="Initial masks directory (default: <data_dir>/masks_initial)")
    parser.add_argument("--output", type=Path, default=None,
                       help="Output masks directory (default: <data_dir>/masks_corrected)")

    args = parser.parse_args()

    # Resolve paths - animation flag takes precedence
    if args.animation:
        # Canonical animation directory structure
        anim_dir = Path("training_data/animations") / args.animation
        if not anim_dir.exists():
            parser.error(f"Animation directory not found: {anim_dir}")

        frames_dir = anim_dir / "frames"
        # For animation mode, read AND write to the same masks/ directory
        initial_masks_dir = anim_dir / "masks"
        corrected_masks_dir = anim_dir / "masks"  # SAME directory - corrections in place

        print(f"Animation mode: {args.animation}")
        print(f"  Frames: {frames_dir}")
        print(f"  Masks (read & write): {initial_masks_dir}")
    elif args.data_dir:
        base = args.data_dir
        frames_dir = args.frames or (base / "frames")
        initial_masks_dir = args.initial or (base / "masks_initial")
        corrected_masks_dir = args.output or (base / "masks_corrected")
    else:
        base = Path("training_data")
        frames_dir = args.frames or (base / "frames")
        initial_masks_dir = args.initial or (base / "masks_initial")
        corrected_masks_dir = args.output or (base / "masks_corrected")

    # Validate directories exist
    if not frames_dir.exists():
        parser.error(f"Frames directory not found: {frames_dir}")
    if not initial_masks_dir.exists():
        parser.error(f"Initial masks directory not found: {initial_masks_dir}")

    correct_all_masks(frames_dir, initial_masks_dir, corrected_masks_dir)
