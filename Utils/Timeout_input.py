import threading

class TimeoutInput:
    def __init__(self, prompt, timeout, default_var):
        self.prompt = prompt
        self.timeout = timeout
        self.default_var = default_var
        self.user_input = None

    def get_input(self):
        self.user_input = input(self.prompt)

    def run(self):
        thread = threading.Thread(target=self.get_input)
        thread.start()
        thread.join(self.timeout)
        if thread.is_alive():
            print("\nTimeout!")
            return {"user_input": self.default_var, "input_time": self.timeout, "default_var_used": True}
        else:
            return {"user_input": self.user_input, "input_time": self.timeout, "default_var_used": False}
