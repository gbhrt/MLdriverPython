import socket

class Comm:
    def __init__(self):
        self.input_data_str = ""
        self.output_data_str = ""
        self.addr = 0
        self.sock = 0
        self.conn = 0
        self.buffSize = 1000000 #1Mb

    def waitForClient(self):
        UDP_IP = "127.0.0.1"
        UDP_PORT = 5000
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print ("try to bind")
        self.sock.bind((UDP_IP, UDP_PORT))
        print ("binded")
        print ("listen")
        self.sock.listen(1)
        print ("connection found")
        self.conn, self.addr =  self.sock.accept()
        print ("Connection from: " + str(self.addr))
        return

    def connectToServer(self,UDP_IP,UDP_PORT):
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            print(UDP_IP," ", UDP_PORT)
            self.sock.connect((UDP_IP, UDP_PORT))#connect to master program
        except ValueError:
            print("cannot connect to server")

    def readData(self):
       # print("wait for input")
        self.input_data_str = self.sock.recv(self.buffSize) 
       # print("input: ",self.input_data_str)
        return

    def sendData(self):
        #self.serialize(999)
       # print("output: ", self.output_data_str)

        self.sock.send(bytes(self.output_data_str, 'UTF-8'))
        self.output_data_str = ""
        return

    def end_connection(self):
        print("end tcp connection - close remote server")
        self.output_data_str = "<END>"
        self.sendData()

    def serialize(self,data):
        try:
            if not isinstance(data, list):#if not a list
                self.output_data_str += str(round(data,3)) # "{:.9f}".format(numvar) str(round(numvar,9))
                self.output_data_str += ",";
            else:
                for val in data:
                    self.output_data_str += str(val)
                    self.output_data_str += ",";
        except ValueError:
            print("serialize error")
        return

    
    def deserialize(self,lenght,type):
        try:
            if lenght > 1:
                data = [0. for _ in range(lenght)]
            else:
                data = 0.
                
            if lenght == 1:#if not a list
                next = self.input_data_str.find(bytes(',', 'UTF-8'))
                tmp = self.input_data_str[0:next];
                self.input_data_str = self.input_data_str[next+1:]
                data = type(tmp);
            else:
                for i in range(lenght):
                    next = self.input_data_str.find(bytes(',', 'UTF-8'))
                    tmp = self.input_data_str[0:next];
                    self.input_data_str = self.input_data_str[next+1:]
                    data[i] = type(tmp);
        except ValueError:
            print("deserialize error")
        return data