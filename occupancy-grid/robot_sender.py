radio.set_group(69)
distance = 0

def on_forever():
    global distance
    distance = Tinybit.Ultrasonic_Car()
    radio.send_value("distance", distance)
    if distance < 20:
        Tinybit.car_ctrl(Tinybit.CarState.CAR_STOP)
        Tinybit.car_ctrl_speed(Tinybit.CarState.CAR_BACK, 70)
        basic.pause(200)
        Tinybit.car_ctrl_speed(Tinybit.CarState.CAR_SPINLEFT, 70)
        basic.pause(200)
    else:
        Tinybit.car_ctrl_speed(Tinybit.CarState.CAR_RUN, 70)
basic.forever(on_forever)
