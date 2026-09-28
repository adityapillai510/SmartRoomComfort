from flask import Flask, render_template, request, redirect, session
import numpy as np
import skfuzzy as fuzz

app = Flask(__name__)
app.secret_key = "smart-room-secret"


# HOME
@app.route("/")
def home():
    return render_template("home.html")


# USER DETAILS
@app.route("/user", methods=["GET", "POST"])
def user():

    if request.method == "POST":

        name = request.form["name"]
        user_type = request.form["user_type"]
        location = request.form["location"]

        # User-entered details are saved
        session["name"] = name
        session["user_type"] = user_type
        session["location"] = location

        return redirect("/input")

    return render_template("user.html")


# ROOM INPUT
@app.route("/input")
def input_page():

    if "name" not in session:
        return redirect("/user")

    return render_template(
        "input.html",
        name=session["name"]
    )


# FUZZY CALCULATION
@app.route("/calculate", methods=["POST"])
def calculate():

    temperature = float(request.form["temperature"])
    humidity = float(request.form["humidity"])

    temp = np.arange(0, 51, 1)
    hum = np.arange(0, 101, 1)
    comfort = np.arange(0, 101, 1)

    # Temperature
    temp_low = fuzz.trimf(temp, [0, 0, 25])
    temp_moderate = fuzz.trimf(temp, [20, 25, 30])
    temp_high = fuzz.trimf(temp, [25, 50, 50])

    # Humidity
    hum_low = fuzz.trimf(hum, [0, 0, 50])
    hum_moderate = fuzz.trimf(hum, [30, 50, 70])
    hum_high = fuzz.trimf(hum, [50, 100, 100])

    # Membership values
    temp_low_value = fuzz.interp_membership(
        temp, temp_low, temperature
    )

    temp_moderate_value = fuzz.interp_membership(
        temp, temp_moderate, temperature
    )

    temp_high_value = fuzz.interp_membership(
        temp, temp_high, temperature
    )

    hum_low_value = fuzz.interp_membership(
        hum, hum_low, humidity
    )

    hum_moderate_value = fuzz.interp_membership(
        hum, hum_moderate, humidity
    )

    hum_high_value = fuzz.interp_membership(
        hum, hum_high, humidity
    )

    # Fuzzy rules
    rule1 = np.fmin(temp_low_value, hum_low_value)
    rule2 = np.fmin(temp_low_value, hum_moderate_value)
    rule3 = np.fmin(temp_moderate_value, hum_low_value)
    rule4 = np.fmin(temp_moderate_value, hum_moderate_value)
    rule5 = np.fmin(temp_moderate_value, hum_high_value)
    rule6 = np.fmin(temp_high_value, hum_low_value)
    rule7 = np.fmin(temp_high_value, hum_moderate_value)
    rule8 = np.fmin(temp_high_value, hum_high_value)
    rule9 = np.fmin(temp_low_value, hum_high_value)

    # Output membership
    comfort_low = fuzz.trimf(comfort, [0, 0, 40])
    comfort_medium = fuzz.trimf(comfort, [30, 50, 70])
    comfort_high = fuzz.trimf(comfort, [60, 100, 100])

    # Rule activation
    low_activation = np.fmax(rule7, rule8)

    medium_activation = np.fmax(
        rule5,
        np.fmax(rule6, rule9)
    )

    high_activation = np.fmax(
        rule1,
        np.fmax(
            rule2,
            np.fmax(rule3, rule4)
        )
    )

    # Apply rules
    low_output = np.fmin(
        low_activation,
        comfort_low
    )

    medium_output = np.fmin(
        medium_activation,
        comfort_medium
    )

    high_output = np.fmin(
        high_activation,
        comfort_high
    )

    # Combine
    aggregated = np.fmax(
        low_output,
        np.fmax(
            medium_output,
            high_output
        )
    )

    # Defuzzification
    if np.sum(aggregated) == 0:
        comfort_score = 50
    else:
        comfort_score = fuzz.defuzz(
            comfort,
            aggregated,
            "centroid"
        )

    comfort_score = round(comfort_score, 2)

    # Comfort level
    if comfort_score < 40:
        comfort_level = "LOW"
    elif comfort_score < 70:
        comfort_level = "MEDIUM"
    else:
        comfort_level = "HIGH"

    # Fan speed
    if comfort_score < 40:
        fan_speed = "HIGH"
    elif comfort_score < 70:
        fan_speed = "MEDIUM"
    else:
        fan_speed = "LOW"

    # IMPORTANT:
    # These values come from the details entered by the user.
    result = {
        "name": session.get("name", ""),
        "user_type": session.get("user_type", ""),
        "location": session.get("location", ""),
        "temperature": temperature,
        "humidity": humidity,
        "score": comfort_score,
        "comfort": comfort_level,
        "fan": fan_speed
    }

    session["result"] = result

    return redirect("/result")


# RESULT
@app.route("/result")
def result_page():

    result = session.get("result")

    if result is None:
        return redirect("/user")

    return render_template(
        "result.html",
        result=result
    )


# ABOUT
@app.route("/about")
def about():
    return render_template("about.html")


if __name__ == "__main__":
    app.run(debug=True)