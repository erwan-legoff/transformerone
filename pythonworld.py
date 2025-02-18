import tensorflow as tf

# Vérifie si TensorFlow fonctionne
print("Hello, TensorFlow!")

# Vérifie la version installée
print("Version de TensorFlow :", tf.__version__)

# Vérifie si un GPU est disponible
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPU détecté : {gpus}")
else:
    print("Aucun GPU détecté, TensorFlow fonctionne sur CPU.")