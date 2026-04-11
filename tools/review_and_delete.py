#!/usr/bin/env python3
"""Quickly review images one by one and delete unwanted ones.

Usage:
    python3 review_and_delete.py --input cropped_161

Key bindings:
    Right / Space / Enter : keep and next
    D / Delete / BackSpace: delete and next
    U                    : undo last delete
    Q / Esc              : quit
"""

from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List

import tkinter as tk
from PIL import Image, ImageTk


SUPPORTED_EXTENSIONS = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}


@dataclass
class DeleteRecord:
    src: Path
    dst: Path


class ImageReviewer:
    def __init__(self, input_dir: Path, recursive: bool, permanent_delete: bool) -> None:
        self.input_dir = input_dir
        self.recursive = recursive
        self.permanent_delete = permanent_delete
        self.deleted_dir = self.input_dir / "_deleted"

        self.images: List[Path] = self._collect_images()
        self.index = 0
        self.kept = 0
        self.deleted = 0
        self.undo_stack: List[DeleteRecord] = []

        self.root = tk.Tk()
        self.root.title("Image Reviewer")
        self.root.geometry("1200x800")

        self.info_var = tk.StringVar(value="")
        self.path_var = tk.StringVar(value="")

        self.info_label = tk.Label(self.root, textvariable=self.info_var, font=("Arial", 14), anchor="w")
        self.info_label.pack(fill="x", padx=12, pady=(12, 4))

        self.path_label = tk.Label(self.root, textvariable=self.path_var, font=("Arial", 10), anchor="w")
        self.path_label.pack(fill="x", padx=12, pady=(0, 8))

        self.image_label = tk.Label(self.root, bg="#111111")
        self.image_label.pack(fill="both", expand=True, padx=12, pady=(0, 12))

        help_text = (
            "Keep: Right/Space/Enter | Delete: D/Delete/BackSpace | Undo: U | Quit: Q/Esc"
        )
        self.help_label = tk.Label(self.root, text=help_text, font=("Arial", 10), anchor="w")
        self.help_label.pack(fill="x", padx=12, pady=(0, 10))

        self.root.bind("<Right>", self.on_keep)
        self.root.bind("<space>", self.on_keep)
        self.root.bind("<Return>", self.on_keep)

        self.root.bind("d", self.on_delete)
        self.root.bind("D", self.on_delete)
        self.root.bind("<Delete>", self.on_delete)
        self.root.bind("<BackSpace>", self.on_delete)

        self.root.bind("u", self.on_undo)
        self.root.bind("U", self.on_undo)

        self.root.bind("q", self.on_quit)
        self.root.bind("Q", self.on_quit)
        self.root.bind("<Escape>", self.on_quit)

        self.root.bind("<Configure>", self.on_resize)

        self._tk_image = None

    def _collect_images(self) -> List[Path]:
        pattern = "**/*" if self.recursive else "*"
        items = sorted(self.input_dir.glob(pattern))
        images = []

        for path in items:
            if not path.is_file():
                continue
            if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            if "_deleted" in path.parts:
                continue
            images.append(path)

        return images

    def run(self) -> None:
        if not self.images:
            print(f"No images found in: {self.input_dir}")
            return

        self.show_current_image()
        self.root.mainloop()

    def show_current_image(self) -> None:
        if self.index >= len(self.images):
            self.info_var.set(
                f"Finished. Total: {len(self.images)} | Kept: {self.kept} | Deleted: {self.deleted}"
            )
            self.path_var.set("All images processed. Press Q or Esc to quit.")
            self.image_label.config(image="", text="DONE", fg="white", bg="#222222", font=("Arial", 42, "bold"))
            return

        current = self.images[self.index]

        try:
            image = Image.open(current)
            image = image.convert("RGB")
        except Exception as exc:
            self.info_var.set(f"Failed to open image: {current.name} ({exc})")
            self.path_var.set(str(current))
            self.index += 1
            self.show_current_image()
            return

        self.info_var.set(
            f"{self.index + 1}/{len(self.images)} | Kept: {self.kept} | Deleted: {self.deleted} | Size: {image.width}x{image.height}"
        )
        self.path_var.set(str(current))

        display_w = max(self.image_label.winfo_width(), 200)
        display_h = max(self.image_label.winfo_height(), 200)
        image.thumbnail((display_w, display_h), Image.Resampling.LANCZOS)

        self._tk_image = ImageTk.PhotoImage(image)
        self.image_label.config(image=self._tk_image, text="")

    def on_resize(self, _event: tk.Event) -> None:
        if self.index < len(self.images):
            self.show_current_image()

    def on_keep(self, _event: tk.Event) -> None:
        if self.index >= len(self.images):
            return

        self.kept += 1
        self.index += 1
        self.show_current_image()

    def _make_deleted_destination(self, src: Path) -> Path:
        relative = src.relative_to(self.input_dir)
        dst = self.deleted_dir / relative
        dst.parent.mkdir(parents=True, exist_ok=True)

        if not dst.exists():
            return dst

        stem = dst.stem
        suffix = dst.suffix
        i = 1
        while True:
            candidate = dst.with_name(f"{stem}_{i}{suffix}")
            if not candidate.exists():
                return candidate
            i += 1

    def on_delete(self, _event: tk.Event) -> None:
        if self.index >= len(self.images):
            return

        current = self.images[self.index]

        try:
            if self.permanent_delete:
                current.unlink(missing_ok=False)
                self.undo_stack.clear()
            else:
                dst = self._make_deleted_destination(current)
                shutil.move(str(current), str(dst))
                self.undo_stack.append(DeleteRecord(src=current, dst=dst))
        except Exception as exc:
            self.info_var.set(f"Delete failed: {exc}")
            return

        self.deleted += 1
        self.index += 1
        self.show_current_image()

    def on_undo(self, _event: tk.Event) -> None:
        if self.permanent_delete:
            self.info_var.set("Undo unavailable in permanent delete mode.")
            return

        if not self.undo_stack:
            self.info_var.set("Nothing to undo.")
            return

        record = self.undo_stack.pop()
        record.src.parent.mkdir(parents=True, exist_ok=True)

        try:
            shutil.move(str(record.dst), str(record.src))
        except Exception as exc:
            self.info_var.set(f"Undo failed: {exc}")
            return

        self.deleted = max(self.deleted - 1, 0)
        self.info_var.set(f"Undo restored: {record.src.name}")

    def on_quit(self, _event: tk.Event) -> None:
        print(f"Total images: {len(self.images)}")
        print(f"Kept: {self.kept}")
        print(f"Deleted: {self.deleted}")

        if not self.permanent_delete:
            print(f"Moved deleted files to: {self.deleted_dir}")

        self.root.destroy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Review images one by one and delete unwanted files quickly.")
    parser.add_argument("--input", required=True, help="Folder containing images.")
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search images recursively in subfolders.",
    )
    parser.add_argument(
        "--permanent-delete",
        action="store_true",
        help="Delete files permanently instead of moving to _deleted folder.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_dir = Path(args.input).expanduser().resolve()
    if not input_dir.exists() or not input_dir.is_dir():
        raise SystemExit(f"Input folder does not exist: {input_dir}")

    reviewer = ImageReviewer(
        input_dir=input_dir,
        recursive=args.recursive,
        permanent_delete=args.permanent_delete,
    )
    reviewer.run()


if __name__ == "__main__":
    main()
