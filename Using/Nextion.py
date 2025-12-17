import serial, time, threading

class Nextion:
    def __init__(self, port="/dev/serial0", baudrate=9600, timeout=1):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.ser = None
        self.lock = threading.Lock()
        self.initialize_serial()
        
    def initialize_serial(self):
        """Инициализация или переинициализация serial порта"""
        try:
            if self.ser and self.ser.is_open:
                self.ser.close()
                time.sleep(0.5)
            
            self.ser = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=self.timeout
            )
            self.ser.dtr = False
            self.ser.rts = False
            time.sleep(1)
            self.ser.reset_input_buffer()
            self.ser.reset_output_buffer()
            print(f"[Nextion] Serial port {self.port} initialized")
            
        except Exception as e:
            print(f"[Nextion ERROR] Failed to initialize serial: {e}")
            self.ser = None

    def send(self, cmd: str, max_retries=2):
        """Отправка команды с защитой от зависания"""
        if not self.ser or not self.ser.is_open:
            print("[Nextion WARN] Serial port not open, reinitializing...")
            self.initialize_serial()
            if not self.ser:
                return False
        
        for retry in range(max_retries):
            try:
                with self.lock:
                    # Проверяем, что порт отвечает
                    if self.ser.in_waiting > 1000:  # Слишком много данных в буфере
                        self.ser.reset_input_buffer()
                    
                    cmd_bytes = cmd.encode('utf-8') + b'\xFF\xFF\xFF'
                    bytes_written = self.ser.write(cmd_bytes)
                    self.ser.flush()
                    
                    print(f"[Nextion] Sent {bytes_written} bytes: {cmd}")
                    time.sleep(0.1)  # Даем время на обработку
                    return True
                    
            except (serial.SerialException, OSError) as e:
                print(f"[Nextion ERROR] Send failed (attempt {retry+1}): {e}")
                if retry == max_retries - 1:
                    print("[Nextion] Reinitializing serial port...")
                    self.initialize_serial()
                time.sleep(0.5)
        
        return False
        
    def _end(self):
        self.ser.write(b'\xFF\xFF\xFF')
        self.ser.flush()  # Добавил очистку буфера

    def _escape_text(self, text: str) -> str:
        if text is None:
            return ""
        # Экранируем кавычки и специальные символы
        return str(text).replace('"', '\\"').replace("'", "\\'")

    def set_text(self, obj: str, text: str):
        t = self._escape_text(text)
        cmd = f'{obj}.txt="{t}"'
        print(f"[Nextion DEBUG] Sending: {cmd}")  # Добавил отладку
        self.send(cmd)

    def set_value(self, obj: str, value: int):
        try:
            val = int(value)
        except:
            val = 0
        cmd = f'{obj}.val={val}'
        self.send(cmd)

    def set_percent(self, obj: str, value: int):
        self.set_value(obj, value)

    def set_visibility(self, obj: str, visible: bool):
        val = 1 if visible else 0
        cmd = f'vis {obj},{val}'
        self.send(cmd)

    def page(self, page_name: str):
        cmd = f'page {page_name}'
        self.send(cmd)
        time.sleep(0.1)  # Даем время на смену страницы

    def __enter__(self):
        return self