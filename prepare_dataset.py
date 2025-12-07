# prepare_dataset.py
import shutil
import os
from pathlib import Path
import cv2
from preprocessing import (
    get_valid_paths, 
    CLEAN_PATH, 
    DIRTY_PATH,
    apply_random_augmentations,
    apply_single_random_augmentation
)

# Tutaj stworzymy folder gotowy dla Anomalib
TARGET_DIR = Path("C:\\Projekty\\Hacknation\\dataset_anomalib")

# 🔥 KONFIGURACJA AUGMENTACJI
ENABLE_AUGMENTATION = True      # Włącz/wyłącz augmentację
AUGMENTATION_MODE = "multiple"  # "single" = 1 losowa aug, "multiple" = kombinacja augs
AUGMENTATION_PROBABILITY = 0.9  # Prawdopodobieństwo każdej augmentacji (tylko dla "multiple")
AUGMENT_TRAIN_ONLY = True       # True = tylko dane treningowe, False = wszystko

def setup_folders():
    train_good = TARGET_DIR / "train" / "good"
    test_bad = TARGET_DIR / "test" / "bad"
    test_good = TARGET_DIR / "test" / "good"

    if TARGET_DIR.exists():
        shutil.rmtree(TARGET_DIR)
    
    train_good.mkdir(parents=True, exist_ok=True)
    test_bad.mkdir(parents=True, exist_ok=True)
    test_good.mkdir(parents=True, exist_ok=True)
    
    return train_good, test_bad, test_good

def copy_files_with_random_augmentation(file_paths, dest_folder, augment=False):
    """
    🔥 NOWA WERSJA: Kopiuje pliki z LOSOWĄ augmentacją dla każdego obrazu
    """
    print(f"\nKopiowanie do {dest_folder.name}...")
    print(f"  Plików: {len(file_paths)}")
    
    if not augment or not ENABLE_AUGMENTATION:
        # Zwykłe kopiowanie bez augmentacji
        for src_path in file_paths:
            new_name = f"{src_path.parent.name}_{src_path.name}"
            shutil.copy(src_path, dest_folder / new_name)
        print(f"  ✓ Skopiowano bez augmentacji: {len(file_paths)} plików")
        return len(file_paths)
    
    # Kopiowanie z LOSOWĄ augmentacją
    print(f"  Augmentacja: {AUGMENTATION_MODE}")
    if AUGMENTATION_MODE == "multiple":
        print(f"  Prawdopodobieństwo: {AUGMENTATION_PROBABILITY}")
    
    aug_stats = {}  # Statystyki jakie augmentacje zostały użyte
    
    for i, src_path in enumerate(file_paths, 1):
        if i % 50 == 0:
            print(f"    Przetworzono {i}/{len(file_paths)}...")
        
        # Wczytaj obraz
        image = cv2.imread(str(src_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            print(f"    ⚠️  Pominięto (błąd wczytania): {src_path.name}")
            continue
        
        # 🔥 Zastosuj LOSOWĄ augmentację
        if AUGMENTATION_MODE == "single":
            # Jedna losowa augmentacja
            augmented, aug_name = apply_single_random_augmentation(image)
            aug_info = aug_name
        else:
            # Kombinacja losowych augmentacji
            augmented, applied_augs = apply_random_augmentations(
                image, 
                probability=AUGMENTATION_PROBABILITY
            )
            aug_info = "_".join(applied_augs)
        
        # Statystyki
        for aug in aug_info.split("_"):
            aug_stats[aug] = aug_stats.get(aug, 0) + 1
        
        # Zapisz z informacją o augmentacji w nazwie
        new_name = f"{src_path.parent.name}_{src_path.stem}_{aug_info}{src_path.suffix}"
        output_path = dest_folder / new_name
        cv2.imwrite(str(output_path), augmented)
    
    print(f"  ✓ Zapisano z losową augmentacją: {len(file_paths)} plików")
    print(f"  📊 Użyte augmentacje:")
    for aug, count in sorted(aug_stats.items(), key=lambda x: x[1], reverse=True):
        print(f"      {aug}: {count}×")
    
    return len(file_paths)

if __name__ == "__main__":
    print("=" * 70)
    print("--- Przygotowanie danych + LOSOWA Data Augmentation ---")
    print("=" * 70)
    
    # Konfiguracja
    print(f"\nKonfiguracja augmentacji:")
    print(f"  Włączona: {ENABLE_AUGMENTATION}")
    print(f"  Tryb: {AUGMENTATION_MODE}")
    if AUGMENTATION_MODE == "multiple":
        print(f"  Prawdopodobieństwo: {AUGMENTATION_PROBABILITY}")
    print(f"  Tylko trening: {AUGMENT_TRAIN_ONLY}")
    
    # 1. Pobierz ścieżki
    clean_paths = get_valid_paths(CLEAN_PATH)
    dirty_paths = get_valid_paths(DIRTY_PATH)
    
    print(f"\nPodsumowanie źródeł:")
    print(f"  Czyste obrazy: {len(clean_paths)} plików")
    print(f"  Brudne obrazy: {len(dirty_paths)} plików")
    
    # 2. Stwórz foldery
    train_good_dir, test_bad_dir, test_good_dir = setup_folders()
    
    # 3. Split danych (80% train, 20% test dla czystych)
    split_idx = int(len(clean_paths) * 0.8)
    clean_train = clean_paths[:split_idx]
    clean_test = clean_paths[split_idx:]
    
    print(f"\nPodział danych:")
    print(f"  Train (good): {len(clean_train)} plików")
    print(f"  Test (good):  {len(clean_test)} plików")
    print(f"  Test (bad):   {len(dirty_paths)} plików")
    
    # 4. Kopiuj pliki z LOSOWĄ augmentacją
    print("\n" + "=" * 70)
    print("--- Kopiowanie z LOSOWĄ augmentacją ---")
    print("=" * 70)
    
    # Czyste do treningu - Z LOSOWĄ AUGMENTACJĄ
    train_total = copy_files_with_random_augmentation(
        clean_train, 
        train_good_dir, 
        augment=AUGMENT_TRAIN_ONLY
    )
    
    # Czyste do testu - BEZ augmentacji
    test_good_total = copy_files_with_random_augmentation(
        clean_test, 
        test_good_dir, 
        augment=False
    )
    
    # Brudne do testu - BEZ augmentacji
    test_bad_total = copy_files_with_random_augmentation(
        dirty_paths, 
        test_bad_dir, 
        augment=False
    )
    
    # 5. Podsumowanie
    print("\n" + "=" * 70)
    print("✓ GOTOWE!")
    print("=" * 70)
    print(f"\nŚcieżka do datasetu: {TARGET_DIR}")
    print(f"\nStatystyki finalne:")
    print(f"  Train/good: {train_total} plików")
    print(f"  Test/good:  {test_good_total} plików")
    print(f"  Test/bad:   {test_bad_total} plików")
    print(f"  RAZEM:      {train_total + test_good_total + test_bad_total} plików")
    
    print("=" * 70)