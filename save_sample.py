import serial
import os

def save_sample(size, save_dir):
    arduino = serial.Serial('COM3', 9600)
    with open(save_dir, 'w') as f:
        if arduino.is_open:
            print(arduino.readline())
            for i in range(size):
                print(str(arduino.readline(), 'utf-8').strip("\r\n"), file=f)
    arduino.close()


if __name__ == '__main__':
    type = 'glass_bottle_coffee_half'
    if not os.path.exists(f'./data/{type}'):
        os.makedirs(f'./data/{type}')
    idx = 0
    for i in range(20):
        save_sample(4000, f'./data/{type}/{idx}.txt')
        print(f'sample saved as ./data/{type}/{idx}.txt')
        idx+=1