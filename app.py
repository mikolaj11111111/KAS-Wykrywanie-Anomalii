import streamlit as st
from sklearn.metrics import f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import torch
import cv2
import numpy as np
from pathlib import Path
from anomalib.models import Padim

st.set_page_config(page_title="Wyszukiwarka Anomalii RTG", layout="wide")

st.title("🔍 Wyszukiwarka Anomalii RTG")

# Sidebar - konfiguracja
with st.sidebar:
    st.header("Konfiguracja")
    
    mode = st.radio(
        "Tryb pracy",
        ["Ewaluacja modelu (batch)", "Pojedyncze zdjęcie"],
        index=0
    )
    
    st.markdown("---")
    
    checkpoint_path = st.text_input(
        "Ścieżka do modelu (.ckpt)",
        value="padim_resnet18_final.ckpt"
    )
    
    if mode == "Ewaluacja modelu (batch)":
        good_files = st.file_uploader(
            "Obrazy dobre (good)",
            type=['bmp'],
            accept_multiple_files=True,
            key='good'
        )
        
        bad_files = st.file_uploader(
            "Obrazy z anomaliami (bad)",
            type=['bmp'],
            accept_multiple_files=True,
            key='bad'
        )
        
        threshold = st.slider("Threshold", 0.0, 1.0, 0.5, 0.01)
        
    else:  # Pojedyncze zdjęcie
        single_file = st.file_uploader(
            "Wybierz obraz RTG",
            type=['bmp'],
            key='single'
        )
        
        threshold = st.slider("Threshold", 0.0, 1.0, 0.5, 0.01)
    
    run_button = st.button("🚀 Uruchom", type="primary")

# Funkcje pomocnicze
@st.cache_resource
def load_model(checkpoint_path):
    """Załaduj model"""
    model = Padim.load_from_checkpoint(checkpoint_path)
    model.eval()
    return model

def process_image(model, image_path, threshold):
    """Przetwórz obraz i zwróć predykcję + heatmapę"""
    # Wczytaj obraz
    uploaded_file.seek(0)
    image = cv2.imread(str(image_path))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Przygotuj tensor
    image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0
    image_tensor = image_tensor.unsqueeze(0)
    
    # Predykcja
    with torch.no_grad():
        output = model(image_tensor)
    
    # Sprawdź typ output i wyciągnij dane
    if isinstance(output, dict):
        # Stary format (dict)
        anomaly_map = output['anomaly_map'][0].cpu().numpy()
        score = output['pred_score'].item()
    else:
        # Nowy format - output jest obiektem z atrybutami
        anomaly_map = output.anomaly_map[0].cpu().numpy()
        score = output.pred_score.item()
    
    prediction = 1 if score > threshold else 0
    
    return {
        'image': image_rgb,
        'anomaly_map': anomaly_map,
        'score': score,
        'prediction': prediction,
        'path': image_path
    }

def process_uploaded_file(model, uploaded_file, threshold):
    """Przetwórz uploadowany obraz i zwróć predykcję + heatmapę"""
    # Wczytaj obraz z bufora
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Przygotuj tensor
    image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0
    image_tensor = image_tensor.unsqueeze(0)
    
    # Predykcja
    with torch.no_grad():
        output = model(image_tensor)
    
    # Sprawdź typ output i wyciągnij dane
    if isinstance(output, dict):
        anomaly_map = output['anomaly_map'][0].cpu().numpy()
        score = output['pred_score'].item()
    else:
        anomaly_map = output.anomaly_map[0].cpu().numpy()
        score = output.pred_score.item()
    
    prediction = 1 if score > threshold else 0
    
    return {
        'image': image_rgb,
        'anomaly_map': anomaly_map,
        'score': score,
        'prediction': prediction,
        'filename': uploaded_file.name
    }

def create_overlay(image, anomaly_map, alpha=0.6):
    """Nałóż heatmapę na obraz"""
    # Upewnij się, że anomaly_map jest 2D
    if len(anomaly_map.shape) == 3:
        # Jeśli 3D, weź pierwszy kanał
        anomaly_map = anomaly_map[0]
    
    # Normalizuj anomaly map
    anomaly_norm = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-8)
    
    # Resize do rozmiaru obrazu
    h, w = image.shape[:2]
    heatmap = cv2.resize(anomaly_norm, (w, h))
    
    # Konwertuj na uint8 (0-255) - musi być 2D, single channel
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    
    # Upewnij się że jest 2D
    if len(heatmap_uint8.shape) > 2:
        heatmap_uint8 = heatmap_uint8[:, :, 0]
    
    # Konwertuj na kolor (jet colormap)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Nałóż na obraz
    overlay = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)
    
    return overlay

# Główna logika
if run_button:
    # Sprawdź czy model istnieje
    if not Path(checkpoint_path).exists():
        st.error(f"❌ Nie znaleziono modelu: {checkpoint_path}")
        st.stop()
    
    # Załaduj model
    with st.spinner("Ładowanie modelu..."):
        model = load_model(checkpoint_path)
    st.success("✅ Model załadowany!")
    
    # === TRYB BATCH - Ewaluacja modelu ===
    if mode == "Ewaluacja modelu (batch)":
        # Walidacja uploadów
        if not good_files and not bad_files:
            st.error("❌ Wgraj przynajmniej jedną grupę obrazów!")
            st.stop()
        
        normal_images = good_files if good_files else []
        anomaly_images = bad_files if bad_files else []
        
        # Przetwórz obrazy
        st.write(f"📊 Przetwarzanie {len(normal_images)} normalnych i {len(anomaly_images)} anomalii...")
        
        results = []
        progress = st.progress(0)
        total = len(normal_images) + len(anomaly_images)
        
        # Normalne obrazy (ground truth = 0)
        for idx, uploaded_file in enumerate(normal_images):
            result = process_uploaded_file(model, uploaded_file, threshold)
            results.append((result, 0))
            progress.progress((idx + 1) / total)
        
        # Anomalie (ground truth = 1)
        for idx, uploaded_file in enumerate(anomaly_images):
            result = process_uploaded_file(model, uploaded_file, threshold)
            results.append((result, 1))
            progress.progress((len(normal_images) + idx + 1) / total)
        
        progress.empty()
        
        # Oblicz metryki
        y_true = [gt for _, gt in results]
        y_pred = [result['prediction'] for result, _ in results]
        y_scores = [result['score'] for result, _ in results]

        accuracy = (sum(1 for p, t in zip(y_pred, y_true) if p == t) / len(results)) * 100
        f1 = f1_score(y_true, y_pred)
        auroc = roc_auc_score(y_true, y_scores)

        # Wyświetl metryki
        st.markdown("---")
        st.header("📊 Wyniki")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Accuracy", f"{accuracy:.1f}%")
        with col2:
            st.metric("F1-Score", f"{f1:.4f}")
        with col3:
            st.metric("AUROC", f"{auroc:.4f}")
        with col4:
            st.metric("Obrazów", f"{len(results)}")

        # Wykres ROC
        st.markdown("---")
        st.header("📈 Krzywa ROC")

        fpr, tpr, thresholds = roc_curve(y_true, y_scores)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auroc:.4f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random classifier')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic (ROC) Curve')
        ax.legend(loc="lower right")
        ax.grid(alpha=0.3)

        st.pyplot(fig)
        
        # Wyświetl anomalie
        st.markdown("---")
        st.header("🔍 Wykryte Anomalie")
        
        anomalies = [(result, gt) for result, gt in results if result['prediction'] == 1]
        
        if not anomalies:
            st.info("Nie wykryto żadnych anomalii")
        else:
            st.write(f"Znaleziono **{len(anomalies)}** anomalii")
            
            for i in range(0, len(anomalies), 2):
                cols = st.columns(2)
                
                for j in range(2):
                    if i + j < len(anomalies):
                        result, gt = anomalies[i + j]
                        
                        with cols[j]:
                            is_correct = result['prediction'] == gt
                            if is_correct and gt == 1:
                                st.success("✅ TRUE POSITIVE - Prawidłowo wykryta anomalia")
                            elif not is_correct:
                                st.warning("⚠️ FALSE POSITIVE - Błędny alarm")
                            
                            st.write(f"**Score:** {result['score']:.4f}")
                            st.write(f"**Plik:** {result['filename']}")
                            
                            tab1, tab2 = st.tabs(["Oryginalny", "Z heatmapą"])
                            
                            with tab1:
                                st.image(result['image'], use_container_width=True)
                            
                            with tab2:
                                overlay = create_overlay(result['image'], result['anomaly_map'])
                                st.image(overlay, use_container_width=True)
                            
                            st.markdown("---")
    
    # === TRYB SINGLE - Pojedyncze zdjęcie ===
    else:
        if not single_file:
            st.error("❌ Wgraj obraz do analizy!")
            st.stop()
        
        # Przetwórz obraz
        with st.spinner("Analizuję obraz..."):
            result = process_uploaded_file(model, single_file, threshold)
        
        # Wyświetl wyniki
        st.markdown("---")
        st.header("📊 Wynik analizy")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Anomaly Score", f"{result['score']:.4f}")
        with col2:
            if result['prediction'] == 1:
                st.error("🚨 ANOMALIA WYKRYTA")
            else:
                st.success("✅ Obraz prawidłowy")
        
        # Wyświetl obrazy
        st.markdown("---")
        
        if result['prediction'] == 1:
            # Anomalia - pokaż oba obrazy w tabach
            tab1, tab2 = st.tabs(["Oryginalny obraz", "Mapa anomalii"])
            
            with tab1:
                st.image(result['image'], use_container_width=True, caption=result['filename'])
            
            with tab2:
                overlay = create_overlay(result['image'], result['anomaly_map'])
                st.image(overlay, use_container_width=True, caption="Heatmapa anomalii")
        else:
            # Brak anomalii - pokaż tylko oryginalny obraz
            st.image(result['image'], use_container_width=True, caption=result['filename'])

else:
    st.info("👈 Ustaw konfigurację w pasku bocznym i kliknij 'Uruchom'")