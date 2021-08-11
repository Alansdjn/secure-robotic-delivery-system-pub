import os
import time
import random
import socket

random.seed(200207124)

def generate_pins(length=4):
    digits = []
    for i in range(length):
        digits += [str(random.randint(0, 9))]
    return digits

print('Server start ...')
time.sleep(1)

pins = generate_pins()
customer_pins = {}

s = socket.socket()
ipaddr = '192.168.31.194'
port = 12345
print(ipaddr, ':', port)
s.bind((ipaddr, port))

s.listen(5)
while True:
    c,addr = s.accept()
    print('Connected address: %s:%s' % addr)
    c.send(','.join(pins).encode('utf-8'))
    print('Send PIN numbers [%s] ...: %s:%s' % (''.join(pins), addr[0], addr[1]))
    c.close()