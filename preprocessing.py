# preprocessor.py
import os
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import random

# Upewnij się, że ścieżki są poprawne!
CLEAN_PATH = "C:\\Projekty\\Hacknation\\HACKATHON-HACKNATION-2025\\NAUKA\\czyste"
DIRTY_PATH = "C:\\Projekty\\Hacknation\\HACKATHON-HACKNATION-2025\\NAUKA\\brudne"

def get_all_bmp_files_from_folder(folder_path: Path) -> list[Path]:
    """Zwraca WSZYSTKIE pliki .bmp z folderu"""
    bmp_files = list(folder_path.glob("*.bmp"))
    return sorted(bmp_files)

def get_valid_paths(base_path: str) -> list[Path]:
    """Zwraca listę WSZYSTKICH ŚCIEŻEK do plików .bmp (zarówno czarno jak i kolor)"""
    base = Path(base_path)
    subfolders = sorted([x for x in base.iterdir() if x.is_dir()])
    
    valid_paths = []
    for subfolder in subfolders:
        all_files = get_all_bmp_files_from_folder(subfolder)
        if all_files:
            valid_paths.extend(all_files)
            
    print(f"[{base.name}] Znaleziono {len(valid_paths)} plików .bmp.")
    return valid_paths


# ========== DATA AUGMENTATION ==========

def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    """Obrót obrazu o podany kąt"""
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, matrix, (w, h), 
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REFLECT)
    return rotated

def flip_image(image: np.ndarray, mode: str = 'horizontal') -> np.ndarray:
    """Odbicie lustrzane"""
    if mode == 'horizontal':
        return cv2.flip(image, 1)
    elif mode == 'vertical':
        return cv2.flip(image, 0)
    elif mode == 'both':
        return cv2.flip(image, -1)
    return image

def add_gaussian_noise(image: np.ndarray, mean: float = 0, sigma: float = 10) -> np.ndarray:
    """Dodaje szum Gaussowski"""
    noise = np.random.normal(mean, sigma, image.shape).astype(np.float32)
    noisy_image = image.astype(np.float32) + noise
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    return noisy_image

def adjust_brightness_contrast(image: np.ndarray, alpha: float = 1.0, beta: float = 0) -> np.ndarray:
    """
    Regulacja jasności i kontrastu
    alpha: kontrast (1.0 = bez zmian, >1 = więcej kontrastu)
    beta: jasność (0 = bez zmian, >0 = jaśniej, <0 = ciemniej)
    """
    adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return adjusted

def apply_random_augmentations(image: np.ndarray, probability: float = 0.9) -> tuple[np.ndarray, list[str]]:
    """
    🔥 NOWA FUNKCJA: Aplikuje LOSOWE augmentacje do obrazu
    
    Args:
        image: obraz wejściowy
        probability: prawdopodobieństwo zastosowania każdej augmentacji (0-1)
    
    Returns:
        (augmentowany obraz, lista zastosowanych augmentacji)
    """
    augmented = image.copy()
    applied_augs = []
    
    # 1. Obrót (losowy kąt od -15 do 15 stopni)
    if random.random() < probability * 0.8:  # 50% szansy na obrót
        angle = random.uniform(-15, 15)
        augmented = rotate_image(augmented, angle)
        applied_augs.append(f'rot{int(angle)}')
    
    # 2. Flip horizontal
    if random.random() < probability * 0.7:  # 30% szansy
        augmented = flip_image(augmented, 'horizontal')
        applied_augs.append('flipH')
    
    # 3. Flip vertical
    if random.random() < probability * 0.85:  # 30% szansy
        augmented = flip_image(augmented, 'vertical')
        applied_augs.append('flipV')
    
    # 4. Szum Gaussowski
    if random.random() < probability * 0.85:  # 40% szansy
        sigma = random.uniform(3, 12)
        augmented = add_gaussian_noise(augmented, sigma=sigma)
        applied_augs.append(f'noise{int(sigma)}')
    
    # 5. Jasność
    if random.random() < probability * 0.85:  # 50% szansy
        beta = random.uniform(-25, 25)
        augmented = adjust_brightness_contrast(augmented, alpha=1.0, beta=beta)
        applied_augs.append(f'bright{int(beta):+d}')
    
    # 6. Kontrast
    if random.random() < probability * 0.85:  # 50% szansy
        alpha = random.uniform(0.7, 1.4)
        augmented = adjust_brightness_contrast(augmented, alpha=alpha, beta=0)
        applied_augs.append(f'cont{alpha:.2f}')
    
    # Jeśli nic nie zostało zastosowane, zwróć oryginał
    if not applied_augs:
        applied_augs.append('orig')
    
    return augmented, applied_augs


def apply_single_random_augmentation(image: np.ndarray) -> tuple[np.ndarray, str]:
    """
    🔥 Aplikuje JEDNĄ losową augmentację
    Użyj tej funkcji jeśli chcesz tylko 1 transformację na obraz
    """
    augmentation_funcs = [
        ('rotate_small', lambda img: (rotate_image(img, random.uniform(-5, 5)), 'rot_sm')),
        ('rotate_medium', lambda img: (rotate_image(img, random.uniform(-15, 15)), 'rot_md')),
        ('flip_h', lambda img: (flip_image(img, 'horizontal'), 'flipH')),
        ('flip_v', lambda img: (flip_image(img, 'vertical'), 'flipV')),
        ('noise_light', lambda img: (add_gaussian_noise(img, sigma=5), 'noise5')),
        ('noise_medium', lambda img: (add_gaussian_noise(img, sigma=12), 'noise12')),
        ('brightness_up', lambda img: (adjust_brightness_contrast(img, alpha=1.0, beta=20), 'bright+20')),
        ('brightness_down', lambda img: (adjust_brightness_contrast(img, alpha=1.0, beta=-20), 'bright-20')),
        ('contrast_up', lambda img: (adjust_brightness_contrast(img, alpha=1.3, beta=0), 'cont1.3')),
        ('contrast_down', lambda img: (adjust_brightness_contrast(img, alpha=0.7, beta=0), 'cont0.7')),
    ]
    
    # Losuj jedną augmentację
    _, aug_func = random.choice(augmentation_funcs)
    augmented, aug_name = aug_func(image.copy())
    
    return augmented, aug_name