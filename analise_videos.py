import os
import cv2
import face_recognition
from deepface import DeepFace
from google.cloud import videointelligence_v1 as vi

# função para analisar e processar o vídeo em questão
def analyze_video(video_path):
    # Inicializa o cliente da API Video Intelligence
    client = vi.VideoIntelligenceServiceClient()

    # Lê o conteúdo do vídeo
    with open(video_path, 'rb') as file:
        input_content = file.read()

    # Configura a solicitação para detecção de rótulos
    features = [vi.Feature.LABEL_DETECTION]
    operation = client.annotate_video(
        request={"features": features, "input_content": input_content}
    )

    print("Processando o vídeo...")
    result = operation.result(timeout=300)
    print("Processamento concluído.")

    # Processa e retorna os resultados
    annotation_results = result.annotation_results[0]
    activities = []
    for label in annotation_results.shot_label_annotations:
        for segment in label.segments:
            start_time = segment.segment.start_time_offset.total_seconds()
            end_time = segment.segment.end_time_offset.total_seconds()
            confidence = segment.confidence
            if confidence > 0.5:  # Filtra atividades com confiança acima de 50%
                activities.append({
                    'description': label.entity.description,
                    'start_time': start_time,
                    'end_time': end_time
                })
    return activities

def add_labels_and_recognition_to_video(input_video_path, output_video_path, activities):
    # Abre o vídeo de entrada
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("Erro ao abrir o vídeo.")
        return

    # Obtém propriedades do vídeo
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Define o codec e cria o objeto VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec para MP4
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Dicionário para armazenar emoções detectadas
    emotions_detected = {}

    # Processa cada frame do vídeo
    frame_number = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Redimensiona o frame para acelerar o processamento
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Detecta rostos no frame atual
        face_locations = face_recognition.face_locations(rgb_small_frame)

        # Calcula o tempo atual em segundos
        current_time = frame_number / fps

        # Verifica se há alguma atividade para o tempo atual
        current_activities = [act['description'] for act in activities
                              if act['start_time'] <= current_time <= act['end_time']]

        # Adiciona o texto das atividades no frame
        if current_activities:
            text = ', '.join(current_activities)
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 255), 2, cv2.LINE_AA)

        # Processa cada rosto detectado
        for (top, right, bottom, left) in face_locations:
            # Redimensiona as coordenadas para o tamanho original
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Extrai a região do rosto
            face_image = frame[top:bottom, left:right]

            try:
                # Analisa as emoções dos rostos detectados
                analyses = DeepFace.analyze(face_image, actions=['emotion'], enforce_detection=False)
                
                # Se apenas uma face for detectada, 'analyses' será um dicionário
                if isinstance(analyses, dict):
                    analyses = [analyses]
                
                for analysis in analyses:
                    emotion = analysis['dominant_emotion']
                    # Atualiza o dicionário de emoções detectadas
                    if emotion not in emotions_detected:
                        emotions_detected[emotion] = 0
                    emotions_detected[emotion] += 1

                    # Desenha o retângulo ao redor do rosto
                    cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)  # Azul em BGR

                    # Desenha o rótulo da emoção abaixo do rosto
                    cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (255, 0, 0), cv2.FILLED)
                    font = cv2.FONT_HERSHEY_DUPLEX
                    cv2.putText(frame, emotion, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
            except Exception as e:
                print(f"Erro na análise de emoção: {e}")


        # Escreve o frame no vídeo de saída
        out.write(frame)
        frame_number += 1

    # Libera os recursos
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    return emotions_detected

def generate_report(activities, emotions_detected, report_path):
    with open(report_path, 'w') as report_file:
        report_file.write("Relatório de Atividades e Emoções Detectadas\n")
        report_file.write("="*50 + "\n\n")

        report_file.write("Atividades Detectadas:\n")
        for activity in activities:
            report_file.write(f"- {activity['description']} (Início: {activity['start_time']}s, Fim: {activity['end_time']}s)\n")
        report_file.write("\n")

        report_file.write("Emoções Detect::contentReference[oaicite:1]{index=1}")
 
 
# Define o caminho para o arquivo de vídeo
video_file_path = "video.mp4"
output_video_path = "video_com_atividades_e_reconhecimento.mp4"
report_file_path = "relatorio_atividades_emocoes.txt" 

# Define a variável de ambiente para autenticação
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "credencial_google.json"

# Analisa o vídeo e obtém as atividades detectadas
atividades_detectadas = analyze_video(video_file_path)

# Adiciona os rótulos de atividades e reconhecimento facial ao vídeo
emocoes_detectadas = add_labels_and_recognition_to_video(
    video_file_path, output_video_path, atividades_detectadas
)

# Gera o relatório com as atividades e emoções detectadas
generate_report(atividades_detectadas, emocoes_detectadas, report_file_path)
