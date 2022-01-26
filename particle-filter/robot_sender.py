radio.set_group(69)
sensor = 0
state = 0
def on_forever():
    global sensor
    sensor = Tinybit.Ultrasonic_Car()
    if sensor < 20:
        Tinybit.car_ctrl(Tinybit.CarState.CAR_STOP)
        Tinybit.car_ctrl_speed(Tinybit.CarState.CAR_BACK, 70)
        basic.pause(100)
        state = 1
        radio.send_value("distance",-70)
        radio.send_value("angle",0)
    elif state == 0:
        Tinybit.car_ctrl_speed(Tinybit.CarState.CAR_RUN, 70)
        radio.send_value("distance",70)
        radio.send_value("angle",0)
    else:
        Tinybit.car_ctrl_speed(Tinybit.CarState.CAR_SPINLEFT, 70)
        basic.pause(100)
        state = 0
        radio.send_value("distance",0)
        radio.send_value("angle",70)
    radio.send_value("sensor", sensor)
basic.forever(on_forever)
