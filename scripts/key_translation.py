import rospy
from pidrone_pkg.msg import Mode
import getch
import time


def main():
    rospy.init_node("key_translation")
    import sys, tty, termios
    fd = sys.stdin.fileno()
    #attr = termios.tcgetattr(sys.stdin.fileno())
    #tty.setraw(sys.stdin.fileno())
    mode = Mode()
    mode.mode = 4
    modepub = rospy.Publisher('/pidrone/set_mode', Mode, queue_size=1)
    rate = rospy.Rate(10)
    msg = """
Commands: 
;:  arm
' ' (spacebar):  disarm
t:  takeoff
l:  land
a:  yaw left
d:  yaw right
w:  up
z:  down
i:  forward
k:  backward
j:  left
l:  right
h:  hover
q:  quit
"""
    try:
        print msg
        while not rospy.is_shutdown():
            ch = getch.getch(0.1)
            if ch == None:
                continue

            if ord(ch) == 32:
                # disarm
                print "disarming"
                mode.mode = 4
                mode.x_velocity = 0
                mode.y_velocity = 0
                mode.z_velocity = 0
                mode.yaw_velocity = 0
                modepub.publish(mode)
            elif ch == ";":
                # arm
                mode.mode = 0                
                mode.x_velocity = 0
                mode.y_velocity = 0
                mode.z_velocity = 0
                mode.yaw_velocity = 0
                modepub.publish(mode)
            elif ch == "y":
                # land
                mode.x_velocity = 0
                mode.y_velocity = 0
                mode.z_velocity = 0
                mode.yaw_velocity = 0
                mode.mode = 3
                modepub.publish(mode)
            elif ch == "h":
                # hover
                mode.mode = 5                
                mode.x_velocity = 0
                mode.y_velocity = 0
                mode.z_velocity = 0
                mode.yaw_velocity = 0
                modepub.publish(mode)
            elif ch == "t":
                mode.x_velocity = 0
                mode.y_velocity = 0
                mode.z_velocity = 0
                mode.yaw_velocity = 0
                mode.mode = 2
                modepub.publish(mode)
            elif ch == "j":
                print "left"
                mode.mode = 5
                mode.x_velocity = -5
                mode.y_velocity = 0
                mode.z_velocity = 0
                mode.yaw_velocity = 0
                modepub.publish(mode)
            elif ch == "l":
                print "right"
                mode.mode = 5
                mode.x_velocity = 5
                mode.y_velocity = 0
                mode.z_velocity = 0
                mode.yaw_velocity = 0
                modepub.publish(mode)
            elif ch == "k":
                print "backward"
                mode.mode = 5
                mode.x_velocity = 0
                mode.y_velocity = -5
                mode.z_velocity = 0
                mode.yaw_velocity = 0
                modepub.publish(mode)
            elif ch == "i":
                print "forward"
                mode.mode = 5
                mode.x_velocity = 0
                mode.y_velocity = 5
                mode.z_velocity = 0
                mode.yaw_velocity = 0
                modepub.publish(mode)
            elif ch == "a":
                print "yaw left"                
                mode.mode = 5
                mode.x_velocity = 0
                mode.y_velocity = 0
                mode.z_velocity = 0
                mode.yaw_velocity = -50
                modepub.publish(mode)
            elif ch == "d":
                print "yaw right"                
                mode.mode = 5
                mode.x_velocity = 0
                mode.y_velocity = 0
                mode.z_velocity = 0
                mode.yaw_velocity = 50
                modepub.publish(mode)
            elif ch == "w":
                print "up"
                mode.mode = 5
                mode.x_velocity = 0
                mode.y_velocity = 0
                mode.z_velocity = 1
                mode.yaw_velocity = 0
                modepub.publish(mode)
            elif ch == "s":
                print "down"
                mode.mode = 5
                mode.x_velocity = 0
                mode.y_velocity = 0
                mode.z_velocity = -1
                mode.yaw_velocity = 0
                modepub.publish(mode)                                
            elif ch == "0":
                mode.mode = 0
            elif ch == "1":
                mode.mode = 1
            elif ch == "2":
                mode.mode = 2
            elif ch == "3":
                mode.mode = 3
            elif ch == "4":
                mode.mode = 4
            elif ch == "5":
                mode.mode = 5
            elif ch == "y":
                mode.mode = 1
            elif ch == "q":
                break
            elif ord(ch) == 3: # Ctrl-C
                break            
            else:
                print "unknown character: '%d'" % ord(ch)
                pass
            print msg
            rate.sleep()
    finally:
        print "sending disarm"
        mode.mode = 4
        modepub.publish(mode)
        time.sleep(0.25)

        
if __name__ == "__main__":
    main()