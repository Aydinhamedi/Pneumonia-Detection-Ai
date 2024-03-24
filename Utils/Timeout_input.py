import threading
import queue
import keyboard

class TimeoutInput:
    """
    A class to get user input with a timeout.

    Attributes:
        prompt (str): The prompt to display to the user.
        timeout (int): The time in seconds to wait for user input.
        default_var (str): The default value to return if the user does not provide input.
        timeout_message (str): The message to display when the input times out.
    """

    def __init__(self, prompt, timeout, default_var, timeout_message="\nTimeout!"):
        self.prompt = prompt
        self.timeout = timeout
        self.default_var = default_var
        self.timeout_message = timeout_message
        self.user_input = None
        self.input_queue = queue.Queue()
        self.stop_thread = False

    def get_input(self):
        """Get user input in a non-blocking manner."""
        print(self.prompt, end='', flush=True)
        while not self.stop_thread:
            if keyboard.is_pressed('\n'):
                line = input()
                if line:
                    self.input_queue.put(line.strip())
                    return

    def run(self):
        """
        Run the TimeoutInput.

        Starts a thread to get user input and waits for the specified timeout.
        If the user does not provide input within the timeout, returns the default value.
        """
        thread = threading.Thread(target=self.get_input)
        thread.start()
        thread.join(self.timeout)
        if thread.is_alive():
            self.stop_thread = True
            print(self.timeout_message)
            return {"user_input": self.default_var, "input_time": self.timeout, "default_var_used": True}
        else:
            self.user_input = self.input_queue.get()
            return {"user_input": self.user_input, "input_time": self.timeout, "default_var_used": False}

# Example usage
if __name__ == "__main__":
    timeout_input = TimeoutInput("Enter something: ", 5, "default", "\nTimeout from TimeoutInput!")
    result = timeout_input.run()
    print(result)
