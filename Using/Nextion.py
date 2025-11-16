import serial
import time

class Nextion:
    def __init__(self, port="/dev/serial0", baudrate=9600, timeout=1):
        self.ser = serial.Serial(port, baudrate, timeout=timeout)
        time.sleep(0.3)

    def _end(self):
        """Отправляет обязательные три байта конца команды."""
        self.ser.write(b'\xFF\xFF\xFF')

    def send(self, cmd: str):
        """Отправляет любую команду Nextion."""
        self.ser.write(cmd.encode('utf-8'))
        self._end()

    def set_text(self, obj: str, text: str):
        """Установить текстовый элемент, например t0.txt."""
        cmd = f'{obj}.txt="{text}"'
        self.send(cmd)

    def set_value(self, obj: str, value: int):
        """Установить числовое значение, например n0.val."""
        cmd = f'{obj}.val={value}'
        self.send(cmd)

    def set_visibility(self, obj: str, visible: bool):
        """Показать/спрятать объект."""
        val = 1 if visible else 0
        cmd = f'vis {obj},{val}'
        self.send(cmd)

    def page(self, page_name: str):
        """Переключить страницу."""
        cmd = f'page {page_name}'
        self.send(cmd)

    def close(self):
        self.ser.close()
