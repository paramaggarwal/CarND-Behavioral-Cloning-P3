import React, { Component } from 'react';
import socketIO from 'socket.io-client';
import logo from './logo.svg';
import './App.css';

class App extends Component {
  constructor(props) {
    super(props)
    this.state = {
      telemetry: {},
      prediction: {},
      // commands: {}
    }
  }

  componentDidMount() {
    this._socket = socketIO('ws://localhost:4567')

    // {throttle: "1.0", steering_angle: "0.5598365068435669"}
    // this._socket.on('steer', (commands) => {
    //   this.setState({ commands })
    //   console.log(commands)
    // });

    // Object {throttle: "1.0000", speed: "13.2788", steering_angle: "1.4464", image: "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQ…JwRwDkc0+CIo/2q8huTHH8scsMgAXjIycYyBnjjj86Bn/2Q=="}
    this._socket.on('telemetry', (telemetry) => {
      this.setState({ telemetry })
      // console.log(telemetry)
    })

    this._socket.on('prediction', (prediction) => {
      this.setState({ prediction })
      console.log(prediction)
    })

  }

  render() {
    return (
      <div className="App">
        {/*<div className="App-header">
          <img src={logo} className="App-logo" alt="logo" />
          <h2>Behavioral Cloning Dashboard</h2>
        </div>*/}

        {/* <p className="App-intro">
          {this.state.telemetry.image ? <img src={"data:image/jpeg;charset=utf-8;base64, " + this.state.telemetry.image} /> : null}
        </p> */}

        {this.state.prediction.layerData && this.state.prediction.layerData.map((layerImages) => {
          return (
            <p>
              {layerImages.map((layerImage) => {
                return (
                    <img width={64} height={64} src={"data:image/jpeg;charset=utf-8;base64, " + layerImage} />
                )
              })}
            </p>
          )
        })}

        <p className="App-intro">
          Steering Angle: {this.state.telemetry.steering_angle}
        </p>

        <p className="App-intro">
          Speed: {this.state.telemetry.speed}
        </p>

        <p className="App-intro">
          Throttle: {this.state.telemetry.throttle}
        </p>
      </div>
    );
  }
}

export default App;
