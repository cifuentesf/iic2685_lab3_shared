#!/usr/bin/env python3
"""
Script para visualizar el campo de verosimilitud del modelo de sensor
para diferentes posiciones del robot
"""
import rclpy
from iic2685_lab3.sensor_model import LikelihoodFieldsSensorModel
import matplotlib.pyplot as plt
import numpy as np


def main():
    rclpy.init()
    
    # Crear nodo del modelo de sensor
    sensor_model = LikelihoodFieldsSensorModel()
    
    # Esperar un poco para que se cargue el mapa
    import time
    time.sleep(1.0)
    
    # Definir posiciones de prueba para el robot (en metros)
    # Ajustar estas posiciones según el mapa
    test_positions = [
        (1.0, 1.0),    # Esquina inferior izquierda
        (1.5, 2.5),    # Centro
        (2.5, 4.0),    # Parte superior
        (0.5, 3.0),    # Lado izquierdo
    ]
    
    # Crear figura con subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # Subplot 1: Mapa original
    ax = axes[0]
    ax.imshow(sensor_model.map_image, cmap='gray', origin='lower')
    ax.set_title('Mapa Original', fontsize=14)
    ax.set_xlabel('X [píxeles]')
    ax.set_ylabel('Y [píxeles]')
    
    # Subplot 2: Campo de verosimilitud base
    ax = axes[1]
    im = ax.imshow(sensor_model.likelihood_field, cmap='hot', origin='lower')
    ax.set_title('Campo de Verosimilitud Base', fontsize=14)
    ax.set_xlabel('X [píxeles]')
    ax.set_ylabel('Y [píxeles]')
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    # Para cada posición de prueba, calcular y mostrar verosimilitud
    for i, (x, y) in enumerate(test_positions):
        ax = axes[i + 2]
        
        # Crear grid de verosimilitudes para diferentes poses con theta=0
        likelihood_grid = np.zeros_like(sensor_model.likelihood_field)
        
        # Calcular verosimilitud para cada punto del mapa
        for py in range(sensor_model.map_height):
            for px in range(sensor_model.map_width):
                # Convertir píxeles a metros
                mx = px * sensor_model.map_resolution + sensor_model.map_origin.position.x
                my = py * sensor_model.map_resolution + sensor_model.map_origin.position.y
                
                # Solo calcular si es espacio libre
                if sensor_model.map_data[py, px] == 0:
                    # Simular medición desde la posición (mx, my)
                    # Por simplicidad, usar el campo precalculado
                    likelihood_grid[py, px] = sensor_model.likelihood_field[py, px]
        
        # Mostrar el campo de verosimilitud
        im = ax.imshow(likelihood_grid, cmap='hot', origin='lower', alpha=0.8)
        
        # Sobreponer el mapa
        ax.imshow(sensor_model.map_image, cmap='gray', origin='lower', alpha=0.3)
        
        # Marcar la posición del robot
        robot_px = int((x - sensor_model.map_origin.position.x) / sensor_model.map_resolution)
        robot_py = int((y - sensor_model.map_origin.position.y) / sensor_model.map_resolution)
        ax.plot(robot_px, robot_py, 'bo', markersize=10, markeredgecolor='white', markeredgewidth=2)
        
        ax.set_title(f'Robot en ({x:.1f}, {y:.1f}) m', fontsize=14)
        ax.set_xlabel('X [píxeles]')
        ax.set_ylabel('Y [píxeles]')
    
    # Ajustar layout
    plt.tight_layout()
    
    # Guardar figura
    fig.savefig('likelihood_field_visualization.png', dpi=150, bbox_inches='tight')
    print("Visualización guardada como 'likelihood_field_visualization.png'")
    
    # Mostrar
    plt.show()
    
    # Cerrar nodo
    sensor_model.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()