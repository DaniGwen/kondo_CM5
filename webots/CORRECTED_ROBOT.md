# Reuse the corrected robot in a new world

The stock Webots Kondo is unchanged. Adding it from the standard library does NOT include our fixes. Our corrected world contains an expanded robot, so copying that world preserves the fixes. A reusable local model is now provided:

`webots/kondo_RL/protos/CorrectedKondoKHR3HV.proto`

Inside this Webots project's worlds folder, add:

```
EXTERNPROTO "../protos/CorrectedKondoKHR3HV.proto"

CorrectedKondoKHR3HV {
  translation 0 0 0.24
  controller "kondo_tcp_server"
  supervisor TRUE
}
```

Working example: `webots/kondo_RL/worlds/kondo-corrected-proto-example.wbt`. It includes the box slab with its top at z=-0.1 and normal gravity/collisions. The 0.24 robot height matches that floor setup; another floor height needs a matching initial/reset height.

This local PROTO includes both thigh pitch axes, both knee axes and right ankle pitch corrected to Y, consistent neutral joint transforms, position sensors, original upper-body limits and +25% leg torque. Hip yaw, hip roll and ankle roll retain their verified axes. It exposes translation, rotation, name, controller, supervisor and selfCollision.

The robot model does not carry floor friction, the slab, WorldInfo timestep, Python dependencies or reward settings. The example world and training launchers configure those. For a different Webots project, copy the PROTO and kondo_tcp_server controller folder, or reference them with suitable paths; keep the absolute versioned mesh/texture URLs accessible. Use 16 ms basicTimeStep and the matching observation-version environment variable. The TCP controller supports internal PROTO joint lookup as well as expanded Robot nodes.

Do not import old replay into the corrected model. Real hardware servo IDs, travel limits and torque calibration remain separate from simulator verification. Asset copyright/license headers from the Cyberbotics model are retained.
