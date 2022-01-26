radio.set_group(69)
def on_received_value(name, value):
    serial.write_line(str(value))
radio.on_received_value(on_received_value)
radio.set_transmit_power(6)