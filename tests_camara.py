import cv2

def main():
    # Abre la cámara (usualmente el índice 0 es la cámara integrada)
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Error: No se puede acceder a la cámara.")
        return

    print("Presiona 'q' para salir.")

    while True:
        # Captura el video cuadro por cuadro
        ret, frame = camera.read()

        if not ret:
            print("Error: No se pudo capturar el cuadro.")
            break

        # Muestra el cuadro capturado
        cv2.imshow('Cámara', frame)

        # Presiona 'q' para salir
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Libera la cámara y cierra las ventanas
    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
