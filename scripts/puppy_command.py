import rospy
from sensor.msg import Led
from puppy_control.msg import Velocity

# --- Colores del LED RGB para indicar estado actual ---
colors = {
    "red":(0, 0, 255),      # STOP
    "green":(0, 255, 0),    # FOLLOWING
    "blue":(255, 0, 0),     # IDLE
    "purple":(255, 0, 188), # HI
    "off":(0 ,0 ,0)
}

# --- Función para ejecutar un ActionGroup preprogramado ---
def run_action(client, action_name, wait=False):
    
    """
    Ejecutar acción preprogramada en el PuppyPi.

    Args:
        client: Service client suscrito a /puppy_control/runActionGroup
        action_name: Nombre de la acción como aparece en "home/pi/PuppyPi_PC_Software/ActionGroups"
        wait: Bloquear la ejecución hasta que termine la acción
    Returns:
        response.success: Flag de ejecución exitosa de la acción
    """
    response = client(f"{action_name}.d6ac",wait)

    try:
        if response.success:
            print(f"Acción '{action_name}' ejecutada: {response.message}")
        else:
            rospy.logwarn(f"Acción {action_name} falló: {response.message}")
        return response.success
    
    except rospy.ServiceException as e:
        rospy.logerr(f"Error llamando servicio: {e}")
        return False

# --- Función para publicar las velocidades de control ---
def publish_vel(pub, vx, vyaw, debug=False):
    """
    Publica los comandos de velocidad al PuppyPi

    Args:
        pub: Nodo publisher (debe estar suscrito a "/puppy_control/velocity/autogait")
        vx: Comando de velocidad de avance (cm/s)
        vyaw: Comando de velocidad de giro en yaw (rad/s)
        debug: Visualizar comandos publicados
    """
    vel = Velocity()
    vel.x = vx
    vel.y = 0.0
    vel.yaw_rate = vyaw
    pub.publish(vel)
    
    if debug:
        print(f"Velocidades publicadas: v = {vx:.2f} m/s, ω = {vyaw:.2f} rad/s.")

# --- Función para detener al robot ---
def stop(pub, client):
    """
    Detiene al robot y lo manda a Home
    
    Args:
        pub: Nodo publisher (debe estar suscrito a "/puppy_control/velocity/autogait")
        client: Service client suscrito a "/puppy_control/go_home"
    """
    publish_vel(pub, 0.0, 0.0)

    try:
        go_home(client)
    except rospy.ServiceException as e:
        rospy.logerr(f"No se pudo ir a home: {e}")

# --- Función para llevar el robot a Home
def go_home(client):
    """
    Devuelve al PuppyPi a su posición de Home
    
    Args:
        client: Service client suscrito a "/puppy_control/go_home"
    """
    client()

# --- Cambiar el color del LED RGB al cambiar de estado ---
def color_state(pub, color):
    """
    Cambia el color de los LED RGB incorporados en la placa del PuppyPi al cambiar de estado.

    Args:
        pub: Nodo publisher (debe estar suscrito a "/sensor/rgb_led")
        color: Lista en formato (r,g,b) del color a publicar
    """
    led = Led()
    led.index = 0
    led.rgb.r = color[2]
    led.rgb.g = color[1]
    led.rgb.b = color[0]
    pub.publish(led)
    rospy.sleep(0.005)
    led.index = 1
    pub.publish(led)