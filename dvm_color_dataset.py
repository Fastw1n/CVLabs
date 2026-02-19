# src/datasets/dvm_color_dataset.py
# PyTorch Dataset для Confirmed_fronts.zip (DVM-CAR фронтальные виды)
# Ожидаем структуру:
# confirmed_fronts/
#   Audi/2008/Brand$$Model$$Year$$Color$$...jpg
#   BMW/...
#
# Метка цвета берётся из имени файла: 4-й токен при split("$$") -> Color

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset

try:
    from sklearn.model_selection import train_test_split
except ImportError as e:
    raise ImportError("Нужно установить scikit-learn: pip install scikit-learn") from e


IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def _is_image(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXTS


def parse_color_from_filename(filename: str) -> str:
    """
    Пример имени:
      Chrysler$$Grand Voyager$$2000$$Blue$$17_4$$4$$image_2.jpg

    Берём 4-й элемент (индекс 3) после split("$$") -> 'Blue'
    """
    stem = Path(filename).name  # с расширением
    parts = stem.split("$$")
    if len(parts) < 4:
        raise ValueError(f"Не могу распарсить цвет: '{filename}' (split('$$') дал {len(parts)} частей)")
    color = parts[3].strip()
    if not color:
        raise ValueError(f"Пустая метка цвета в имени: '{filename}'")
    return color


def build_samples(root_dir: str | Path) -> List[Tuple[str, str]]:
    """
    Возвращает список (path_to_image, color_label)
    """
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"Папка не найдена: {root}")

    samples: List[Tuple[str, str]] = []
    for p in root.rglob("*"):
        if p.is_file() and _is_image(p):
            try:
                color = parse_color_from_filename(p.name)
            except ValueError:
                # если вдруг встретится файл с нестандартным именем — пропустим
                continue
            samples.append((str(p), color))

    if not samples:
        raise RuntimeError(
            f"Не нашёл изображений в {root}. Проверь, что распаковал Confirmed_fronts.zip и указал верный путь."
        )
    return samples


@dataclass
class SplitResult:
    train: List[Tuple[str, str]]
    val: List[Tuple[str, str]]
    test: List[Tuple[str, str]]
    class_to_idx: Dict[str, int]
    idx_to_class: Dict[int, str]


def make_splits(
    samples: Sequence[Tuple[str, str]],
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42,
    min_count_per_class: int = 2,
) -> SplitResult:
    """
    Стратифицированные сплиты по цвету.
    Если какие-то классы слишком редкие, их можно:
      - выкинуть (по умолчанию: если < min_count_per_class)
    """
    # фильтр редких классов
    from collections import Counter

    counts = Counter([y for _, y in samples])
    filtered = [(x, y) for x, y in samples if counts[y] >= min_count_per_class]

    if len(filtered) < len(samples):
        removed = len(samples) - len(filtered)
        print(f"[make_splits] Удалил {removed} примеров из слишком редких классов (<{min_count_per_class}).")

    X = [x for x, _ in filtered]
    y = [lab for _, lab in filtered]

    # сначала отделим test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=seed,
        stratify=y
    )

    # затем val из оставшегося trainval
    val_ratio_in_trainval = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=val_ratio_in_trainval,
        random_state=seed,
        stratify=y_trainval
    )

    # классы фиксируем по train (правильнее)
    classes = sorted(set(y_train))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    idx_to_class = {i: c for c, i in class_to_idx.items()}

    train = list(zip(X_train, y_train))
    val = list(zip(X_val, y_val))
    test = list(zip(X_test, y_test))

    return SplitResult(train=train, val=val, test=test, class_to_idx=class_to_idx, idx_to_class=idx_to_class)


class DVMColorDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[Tuple[str, str]],
        class_to_idx: Dict[str, int],
        transform: Optional[Callable] = None,
    ):
        self.samples = list(samples)
        self.class_to_idx = dict(class_to_idx)
        self.transform = transform

        # проверим что все метки есть в словаре
        missing = {y for _, y in self.samples if y not in self.class_to_idx}
        if missing:
            raise ValueError(f"В samples есть метки, которых нет в class_to_idx: {sorted(missing)[:20]}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label_str = self.samples[idx]
        y = self.class_to_idx[label_str]

        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        return img, torch.tensor(y, dtype=torch.long)
