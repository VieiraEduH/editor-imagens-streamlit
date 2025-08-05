import streamlit as st
import cv2 as cv
import numpy as np
from PIL import Image
from io import BytesIO

#Funções de transformações
def img_rotacao(img, angle):
    rows, cols = img.shape[:2] # Obtém as dimensões da imagem
    center = ((cols - 1) / 2, (rows - 1) / 2) # Centro da imagem
    M = cv.getRotationMatrix2D(center, angle, 1.0) # Matriz de rotação
    return cv.warpAffine(img, M, (cols, rows)) # Aplica a rotação

@st.cache_data
def img_escala(img, fx, fy):
    return cv.resize(img, None, fx=fx, fy=fy, interpolation=cv.INTER_LINEAR)

@st.cache_data
def img_shear(img, shx, shy):
    rows, cols = img.shape[:2] # Obtém as dimensões da imagem
    M = np.float32([[1, shx, 0], [shy, 1, 0]]) # Matriz de cisalhamento
    return cv.warpAffine(img, M, (cols, rows)) # Aplica a cisalhamento

@st.cache_data
def img_brilho(img, beta):
    return cv.convertScaleAbs(img, alpha=1.0, beta=beta) # Ajusta o brilho

@st.cache_data
def img_contraste(img, alpha):
    return cv.convertScaleAbs(img, alpha=alpha, beta=0) # Ajusta o contraste

@st.cache_data
def img_gamma(img, gamma):
    inv = 1.0 / gamma # Calcula o inverso do valor gama
    table = np.array([((i / 255.0) ** inv) * 255 for i in np.arange(256)], dtype="uint8") # Cria a tabela de transformação
    return cv.LUT(img, table) # Aplica a transformação gama

@st.cache_data
def img_negativo(img):
    return cv.bitwise_not(img) # Aplica o negativo

st.title("Editor de Imagens")

if 'imagem_carregada' not in st.session_state:
    st.session_state.imagem_carregada = None

# Função para carregar imagem
upload = st.file_uploader("📁 Carregar Imagem", type=["jpg", "jpeg", "png"])
if upload:
    st.session_state.imagem_carregada = upload

# Exibe a imagem carregada
if st.session_state.imagem_carregada:
    img = Image.open(st.session_state.imagem_carregada).convert("RGB") # Converte para RGB
    img_np = np.array(img) # Converte para array NumPy
    img_cv = cv.cvtColor(img_np, cv.COLOR_RGB2BGR) # Converte para BGR (OpenCV)

    with st.container():
        # Exibe a imagem original
        st.subheader("📸 Imagem Original")
        st.image(img, caption="Original", use_container_width =True)

    with st.container():
        # Exibe as opções de transformação
        st.subheader("🔄 Transformações")

        # Controles de transformação
        scale = st.slider("Escala", 1.0, 5.0, 1.0, step=0.1)
        angle = st.slider("Rotação", -180, 180, 0)
        shear_x = st.slider("Cisalhamento Horizontal", -1.0, 1.0, 0.0, step=0.01)
        shear_y = st.slider("Cisalhamento Vertical", -1.0, 1.0, 0.0, step=0.01)
        
        # Cotroles de brilho, contraste, gama e negativo
        beta = st.slider("Brilho", -100, 100, 0)
        alpha = st.slider("Contraste", 0.1, 3.0, 1.0, step=0.1)
        gamma = st.slider("Gama", 0.1, 3.0, 1.0, step=0.1)
        negativo = st.checkbox("Aplicar Negativo")
        
        # Processamentos
        img_processada = img_rotacao(img_cv, angle)
        img_processada = img_escala(img_processada, scale, scale)
        img_processada = img_shear(img_processada, shear_x, shear_y)
        img_processada = img_brilho(img_processada, beta)
        img_processada = img_contraste(img_processada, alpha)
        img_processada = img_gamma(img_processada, gamma)
        if negativo:
            img_processada = img_negativo(img_processada)

        # Converter para exibição
        img_rgb = cv.cvtColor(img_processada, cv.COLOR_BGR2RGB) # Converte de BGR para RGB
        img_final = Image.fromarray(img_rgb) # Converte para PIL para exibição

        st.image(img_final, caption="Imagem transformada", use_container_width =True)
        
        # Botão para download
        st.subheader("💾 Download da Imagem Transformada")            
        buffer = BytesIO()
        img_final.save(buffer, format="PNG")
        st.download_button("💾 Baixar Imagem Transformada", buffer.getvalue(), file_name="imagem_transformada.png", mime="image/png")