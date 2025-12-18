"""GUI for selecting the golden reference frame."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from typing import List


def select_golden_frame(frames: List[np.ndarray]) -> int:
    """Show GUI for user to select the golden reference frame.

    Args:
        frames: List of RGBA images to choose from

    Returns:
        Index of selected golden frame
    """
    if not frames:
        raise ValueError("No frames provided")

    if len(frames) == 1:
        return 0

    n_frames = len(frames)
    selected_idx = [0]  # Use list to allow modification in nested function
    confirmed = [False]

    # Create figure with grid layout
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("Select Golden Frame - Click thumbnail, then 'Select as Golden'", fontsize=14)

    # Left side: 5x5 thumbnail grid
    # Right side: Large preview
    # Bottom: Button

    # Calculate grid dimensions
    grid_cols = 5
    grid_rows = (n_frames + grid_cols - 1) // grid_cols

    # Create axes for thumbnails
    thumb_axes = []
    for i in range(n_frames):
        row = i // grid_cols
        col = i % grid_cols
        # Thumbnails take left 60% of figure, arranged in grid
        ax = fig.add_axes([
            0.02 + col * 0.11,  # x position
            0.85 - row * 0.16,  # y position (top to bottom)
            0.10,  # width
            0.14   # height
        ])
        ax.imshow(frames[i])
        ax.set_title(f"{i:02d}", fontsize=8)
        ax.axis('off')
        thumb_axes.append(ax)

    # Large preview on right
    preview_ax = fig.add_axes([0.60, 0.15, 0.38, 0.75])
    preview_img = preview_ax.imshow(frames[0])
    preview_ax.set_title(f"Frame 00", fontsize=12)
    preview_ax.axis('off')

    # Highlight current selection in thumbnails
    highlights = []
    for ax in thumb_axes:
        rect = plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                             fill=False, edgecolor='yellow', linewidth=3, visible=False)
        ax.add_patch(rect)
        highlights.append(rect)
    highlights[0].set_visible(True)

    # Button
    button_ax = fig.add_axes([0.70, 0.02, 0.20, 0.06])
    button = Button(button_ax, 'Select as Golden')

    def on_thumbnail_click(event):
        if event.inaxes in thumb_axes:
            idx = thumb_axes.index(event.inaxes)
            selected_idx[0] = idx

            # Update preview
            preview_img.set_data(frames[idx])
            preview_ax.set_title(f"Frame {idx:02d}", fontsize=12)

            # Update highlights
            for i, h in enumerate(highlights):
                h.set_visible(i == idx)

            fig.canvas.draw_idle()

    def on_button_click(event):
        confirmed[0] = True
        plt.close(fig)

    fig.canvas.mpl_connect('button_press_event', on_thumbnail_click)
    button.on_clicked(on_button_click)

    # Also support keyboard navigation
    def on_key(event):
        if event.key == 'enter':
            confirmed[0] = True
            plt.close(fig)
        elif event.key == 'left':
            new_idx = max(0, selected_idx[0] - 1)
            selected_idx[0] = new_idx
            preview_img.set_data(frames[new_idx])
            preview_ax.set_title(f"Frame {new_idx:02d}", fontsize=12)
            for i, h in enumerate(highlights):
                h.set_visible(i == new_idx)
            fig.canvas.draw_idle()
        elif event.key == 'right':
            new_idx = min(n_frames - 1, selected_idx[0] + 1)
            selected_idx[0] = new_idx
            preview_img.set_data(frames[new_idx])
            preview_ax.set_title(f"Frame {new_idx:02d}", fontsize=12)
            for i, h in enumerate(highlights):
                h.set_visible(i == new_idx)
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect('key_press_event', on_key)

    plt.show()

    return selected_idx[0]
