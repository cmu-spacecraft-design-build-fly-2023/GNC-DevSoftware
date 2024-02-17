import time
import serial
import json 
import sys
import keyboard

class PyCubedSerial:
    def __init__(self, port, baudrate):
        self.ser = serial.Serial(port, baudrate, timeout=1)

    def send(self, dict_data):
        json_data = json.dumps(dict_data)
        json_data += '\r\n'
        encoded_data = json_data.encode()
        self.ser.write(encoded_data)
        self.ser.flush()

    def read(self):
        curr_line = self.ser.readline()
        curr_line = curr_line.strip()
        return curr_line
    
    def cleanup(self):
        self.ser.close() 
        
if __name__ == "__main__":
    pycubed_serial = PyCubedSerial('COM6', 9600)
    keyboard.on_press_key("q", lambda _: pycubed_serial.cleanup())
    while True:
        pycubed_serial.send({'accel':3.2,'mag':4.5,'gyro':5.6})
        # print(f'[Read] {pycubed_serial.read()}')
        time.sleep(1)
