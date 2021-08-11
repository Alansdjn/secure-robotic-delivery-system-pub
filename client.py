import os
import time
import socket

print('Client start ...')
print('Request PIN numbers from server ...')

ipaddr = '192.168.31.194'
port = 12345
print('Connect to server %s:%s ...' % (ipaddr, port))

s = socket.socket()
s.connect((ipaddr, port))
pins = s.recv(1024).decode()
print('Receive PIN numbers from the server:\n')
print('  '.join(pins.split(',')))

s.close()
