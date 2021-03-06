#The controller class starts a server that get connected to a client in unity.
#So that the controller can control functions in unity out of python
#Copywriter by Felix Loos (felix.loos@t-online.de)

import socket

class Controller:
	sock = None
	address = None

	def __init__(self, ipAddress, port):
		#initialize the socket, the IP-Address and the port
		self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
		self.address = (ipAddress, port)

	def startController(self):
		#Start the server
		self.sock.bind(self.address)

	def sendMessage(self, message):
		#send a message to the predefined address
		self.sock.sendto(message, self.address)

	def recvData(self, bufferSize=2097152):
		#receive data from the socket
		data, addr = self.sock.recvfrom(bufferSize)
		if data is None:
			print("data")
		return data, addr
