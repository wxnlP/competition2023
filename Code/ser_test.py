import serial
import time
import RPi.GPIO as GPIO
ser = serial.Serial("/dev/ttyAMA0",9600)
LED_white = 12
GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)
GPIO.setup(LED_white,GPIO.OUT)

try:
    while True:
        if not ser.isOpen():
            ser.open()
        count = ser.inWaiting()
        print(count)
        if count>=0:
            ser.write(bytes("30", "utf-8"))
            print("开始")
            print("继续")
            respones = ser.read().decode()
            print(f"数据：{respones}")

        
except KeyboardInterrupt:
    ser.close()
    GPIO.cleanup()
    
    
    
    
