#include <math.h>
#include <uWS/uWS.h>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>
#include "Eigen-3.3/Eigen/Core"
#include "Eigen-3.3/Eigen/QR"
#include "MPC.h"
#include "json.hpp"
//#include "matplotlibcpp.h" getting errors on my main Mac. diabling plotting for submission



// for convenience
using json = nlohmann::json;


// For converting back and forth between radians and degrees.
constexpr double pi() { return M_PI; }
double deg2rad(double x) { return x * pi() / 180; }
double rad2deg(double x) { return x * 180 / pi(); }

// Checks if the SocketIO event has JSON data.
// If there is data the JSON object in string format will be returned,
// else the empty string "" will be returned.
string hasData(string s) {
    auto found_null = s.find("null");
    auto b1 = s.find_first_of("[");
    auto b2 = s.rfind("}]");
    if (found_null != string::npos) {
        return "";
    } else if (b1 != string::npos && b2 != string::npos) {
        return s.substr(b1, b2 - b1 + 2);
    }
    return "";
}

// Evaluate a polynomial.
double polyeval(Eigen::VectorXd coeffs, double x) {
    double result = 0.0;
    for (int i = 0; i < coeffs.size(); i++) {
        result += coeffs[i] * pow(x, i);
    }
    return result;
}

// Fit a polynomial.
// Adapted from
// https://github.com/JuliaMath/Polynomials.jl/blob/master/src/Polynomials.jl#L676-L716
Eigen::VectorXd polyfit(Eigen::VectorXd xvals, Eigen::VectorXd yvals,
                        int order) {
    assert(xvals.size() == yvals.size());
    assert(order >= 1 && order <= xvals.size() - 1);
    Eigen::MatrixXd A(xvals.size(), order + 1);
    
    for (int i = 0; i < xvals.size(); i++) {
        A(i, 0) = 1.0;
    }
    
    for (int j = 0; j < xvals.size(); j++) {
        for (int i = 0; i < order; i++) {
            A(j, i + 1) = A(j, i) * xvals(j);
        }
    }
    
    auto Q = A.householderQr();
    auto result = Q.solve(yvals);
    return result;
}

Eigen::MatrixXd convertGlobalToLocalCoord(double carX, double carY, double carPsi, const vector<double> & ptsx, const vector<double> & ptsy){
    assert(ptsx.size() == ptsy.size());
    unsigned len = ptsx.size();
    
    auto waypoints = Eigen::MatrixXd(2,len);
    
    for (unsigned int i=0; i<len ; ++i){
        waypoints(0,i) =   cos(-carPsi) * (ptsx[i] - carX) - sin(-carPsi) * (ptsy[i] - carY);
        waypoints(1,i) =  sin(-carPsi) * (ptsx[i] - carX) + cos(-carPsi) * (ptsy[i] - carY);
    }
    
    return waypoints;
}




int main() {
    uWS::Hub h;
    
    // MPC is initialized here!
    MPC mpc;
    
    
    h.onMessage([&mpc](uWS::WebSocket<uWS::SERVER> ws, char *data, size_t length,
                       uWS::OpCode opCode) {
        
        // "42" at the start of the message means there's a websocket message event.
        // The 4 signifies a websocket message
        // The 2 signifies a websocket event
        string sdata = string(data).substr(0, length);
        cout << sdata << endl;
        if (sdata.size() > 2 && sdata[0] == '4' && sdata[1] == '2') {
            string s = hasData(sdata);
            if (s != "") {
                auto j = json::parse(s);
                string event = j[0].get<string>();
                if (event == "telemetry") {
                    
                    vector<double> ptsx = j[1]["ptsx"];
                    vector<double> ptsy = j[1]["ptsy"];
                    double px = j[1]["x"];
                    double py = j[1]["y"];
                    double psi = j[1]["psi"];
                    double v = j[1]["speed"];
                    double delta = j[1]["steering_angle"];
                    double acceleration = j[1]["throttle"];
                    
                    // Convert  mph -> m/s
                    v=v * 1609.34/3600;
                    
                    double Lf = 2.67;
                    // predict state in 100ms
                    double latency = 0.1;
                    px = px + v*cos(psi)*latency;
                    py = py + v*sin(psi)*latency;
                    psi = psi + v*delta/Lf*latency;
                    v = v + acceleration*latency;
                    
                    // Transform map to car coordinates
                    Eigen::MatrixXd carCoordWaypoints = convertGlobalToLocalCoord(px,py,psi,ptsx,ptsy);
                    Eigen::VectorXd carCoordptsX = carCoordWaypoints.row(0);
                    Eigen::VectorXd carCoordptsY = carCoordWaypoints.row(1);
                    
                    // Try to fit a 3rd order polynomial to represent waypoints path and fetch coefficients
                    Eigen::VectorXd coeffsFit = polyfit(carCoordptsX, carCoordptsY, 3);
                    
                    // Calculate cross track error in reference to the car coordinates. Car is located at (0,0). The new cte is the y value of the polynomial for x=0
                    double cte = polyeval(coeffsFit, 0);
                    
                    // Calculate orientation track error
                    double ote = - atan(coeffsFit[1]);
                    
                    
                    // The state in car's coordinates
                    Eigen::VectorXd state(6);
                    state << 0, 0, 0, v, cte, ote;
                    
                    std::vector<double> x_vals = {state[0]};
                    std::vector<double> y_vals = {state[1]};
                    std::vector<double> psi_vals = {state[2]};
                    std::vector<double> v_vals = {state[3]};
                    std::vector<double> cte_vals = {state[4]};
                    std::vector<double> epsi_vals = {state[5]};
                    std::vector<double> delta_vals = {};
                    std::vector<double> a_vals = {};

                    
                    // Calculate the best trajectory
                    auto bestTr = mpc.Solve(state, coeffsFit);
                    /*
                     * Calculate steering angle and throttle using MPC.
                     *
                     * Both are in between [-1, 1].
                     *
                     */
                    x_vals.push_back(bestTr[0]);
                    y_vals.push_back(bestTr[1]);
                    psi_vals.push_back(bestTr[2]);
                    v_vals.push_back(bestTr[3]);
                    cte_vals.push_back(bestTr[4]);
                    epsi_vals.push_back(bestTr[5]);
                    
                    delta_vals.push_back(bestTr[6]);
                    a_vals.push_back(bestTr[7]);
                    
                    state << bestTr[0], bestTr[1], bestTr[2], bestTr[3], bestTr[4], bestTr[5];
                    std::cout << "x = " << bestTr[0] << std::endl;
                    std::cout << "y = " << bestTr[1] << std::endl;
                    std::cout << "psi = " << bestTr[2] << std::endl;
                    std::cout << "v = " << bestTr[3] << std::endl;
                    std::cout << "cte = " << bestTr[4] << std::endl;
                    std::cout << "epsi = " << bestTr[5] << std::endl;
                    std::cout << "delta = " << bestTr[6] << std::endl;
                    std::cout << "a = " << bestTr[7] << std::endl;
                    std::cout << std::endl;
                    
                    
                    double steer_value = bestTr[0];
                    double throttle_value = bestTr[1];
                    
                    json msgJson;
                    // NOTE: Remember to divide by deg2rad(25) before you send the steering value back.
                    // Otherwise the values will be in between [-deg2rad(25), deg2rad(25] instead of [-1, 1].
                    msgJson["steering_angle"] = -steer_value/(deg2rad(25));
                    
                    msgJson["throttle"] = throttle_value;
                    
                    //Display the MPC predicted trajectory
                    vector<double> mpc_x_vals;
                    vector<double> mpc_y_vals;
                    
                    //          .. add (x,y) points to list here, points are in reference to the vehicle's coordinate system
                    //           the points in the simulator are connected by a Green line
                    
                    for (unsigned int i = 2; i < bestTr.size(); i+=2) {
                        mpc_x_vals.push_back(bestTr[i]);
                        mpc_y_vals.push_back(bestTr[i+1]);
                    }
                    
                    
                    msgJson["mpc_x"] = mpc_x_vals;
                    msgJson["mpc_y"] = mpc_y_vals;
                    
                    //Display the waypoints/reference line
                    vector<double> next_x_vals;
                    vector<double> next_y_vals;
                    
                    //.. add (x,y) points to list here, points are in reference to the vehicle's coordinate system
                    // the points in the simulator are connected by a Yellow line
                    
                    
                    for (unsigned int i=0; i < ptsx.size(); ++i) {
                        next_x_vals.push_back(carCoordptsX(i));
                        next_y_vals.push_back(carCoordptsY(i));
                    }
                    
                    msgJson["next_x"] = next_x_vals;
                    msgJson["next_y"] = next_y_vals;
                    
                    
                    auto msg = "42[\"steer\"," + msgJson.dump() + "]";
                    //std::cout << msg << std::endl; too much data on terminal
                    // Latency
                    // The purpose is to mimic real driving conditions where
                    // the car does actuate the commands instantly.
                    //
                    // Feel free to play around with this value but should be to drive
                    // around the track with 100ms latency.
                    //
                    // NOTE: REMEMBER TO SET THIS TO 100 MILLISECONDS BEFORE
                    // SUBMITTING.
                    this_thread::sleep_for(chrono::milliseconds(100)); // gut feel 100ms is too high in real world
                    ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
                    
                    
                    /* Disabled plot as I am getting complile issues with conflicts of python and openssl
                     Individually each runs good seperately but combined has some issues. Will fix after holidays
                     plt::subplot(3, 1, 1);
                     plt::title("CTE");
                     plt::plot(cte_vals);
                     plt::subplot(3, 1, 2);
                     plt::title("Delta (Radians)");
                     plt::plot(delta_vals);
                     plt::subplot(3, 1, 3);
                     plt::title("Velocity");
                     plt::plot(v_vals);
                     
                     plt::show();
                     **/
                }
            } else {
                // Manual driving
                std::string msg = "42[\"manual\",{}]";
                ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
            }
        }
    });
    
    // We don't need this since we're not using HTTP but if it's removed the
    // program
    // doesn't compile :-(
    h.onHttpRequest([](uWS::HttpResponse *res, uWS::HttpRequest req, char *data,
                       size_t, size_t) {
        const std::string s = "<h1>Hello world!</h1>";
        if (req.getUrl().valueLength == 1) {
            res->end(s.data(), s.length());
        } else {
            // i guess this should be done more gracefully?
            res->end(nullptr, 0);
        }
    });
    
    h.onConnection([&h](uWS::WebSocket<uWS::SERVER> ws, uWS::HttpRequest req) {
        std::cout << "Connected!!!" << std::endl;
    });
    
    h.onDisconnection([&h](uWS::WebSocket<uWS::SERVER> ws, int code,
                           char *message, size_t length) {
        ws.close();
        std::cout << "Disconnected" << std::endl;
    });
    
    int port = 4567;
    if (h.listen(port)) {
        std::cout << "Listening to port " << port << std::endl;
    } else {
        std::cerr << "Failed to listen to port" << std::endl;
        return -1;
    }
    h.run();
}
