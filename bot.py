class Bot:
    """The client that connects to showdown and plays games"""

    def __init__(self, server_url):
        self.server_url = server_url

    def connect(self):
        """Opens a websocket connection with server specified by self.server_url"""

    def disconnect(self):
        """Closes websocket connection with server"""
        pass

    def challenge(self):
        """Challenges a user to a battle"""
        pass

    def ladder(self):
        """Play a certain number of games on the ladder"""


