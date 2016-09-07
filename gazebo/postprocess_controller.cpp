#include <gazebo/gazebo.hh>
#include <gazebo/common/Plugin.hh>
#include <gazebo/common/common.hh>
#include <gazebo/common/Events.hh>
#include <gazebo/physics/physics.hh>

#include "log.h"
#include "models.h"
#include "weazelball.h"

#include <stdlib.h>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>

#define PI 3.14159265359

//-----------------------------------------------------------------------------
// set on the interval [0,9] to switch between different vicon motion capture
// sessions.
unsigned trialid = 9;

//-----------------------------------------------------------------------------
gazebo::math::Vector3 to_omega( gazebo::math::Quaternion q, gazebo::math::Quaternion qd ) {
  gazebo::math::Vector3 omega;
  omega.x = 2 * (-q.x * qd.w + q.w * qd.x - q.z * qd.y + q.y * qd.z);
  omega.y = 2 * (-q.y * qd.w + q.z * qd.x + q.w * qd.y - q.x * qd.z);
  omega.z = 2 * (-q.z * qd.w - q.y * qd.x + q.x * qd.y + q.w * qd.z);
  return omega;

}

//-----------------------------------------------------------------------------
gazebo::math::Quaternion deriv(gazebo::math::Quaternion q, gazebo::math::Vector3 w) {
  gazebo::math::Quaternion qd;

  qd.w = .5 * (-q.x * w.x - q.y * w.y - q.z * w.z); 
  qd.x = .5 * (+q.w * w.x + q.z * w.y - q.y * w.z);
  qd.y = .5 * (-q.z * w.x + q.w * w.y + q.x * w.z);
  qd.z = .5 * (+q.y * w.x - q.x * w.y + q.w * w.z);

  return qd;
}

//-----------------------------------------------------------------------------
namespace gazebo 
{
  class controller_c : public ModelPlugin
  {
  private:
    event::ConnectionPtr _updateConnection;

    world_ptr _world;
    weazelball_ptr _weazelball;

    wb_fused_data_ptr _wb_fused_data;

    log_ptr state_log;
  public:
    //-------------------------------------------------------------------------
    controller_c( void ) { }

    //-------------------------------------------------------------------------
    virtual ~controller_c( void ) {
      //if( log ) log->close();

      event::Events::DisconnectWorldUpdateBegin( _updateConnection );
 
      _world->close();
    }

    //-------------------------------------------------------------------------
    // Gazebo callback.  Called when the simulation is starting up
    virtual void Load( physics::ModelPtr model, sdf::ElementPtr sdf ) {

      std::string validation_errors;
      _world = world_ptr( new world_c( model->GetWorld() ) );
      if( !_world->validate( validation_errors ) ) {
        printf( "Unable to validate world in controller\n%s\nERROR: Plugin failed to load\n", validation_errors.c_str() );
        return;
      }

      // open the world before we begin
      _world->open();

      // get references to objects in the world
      _weazelball = _world->weazelball();
   
      // -- CALLBACKS --
      _updateConnection = event::Events::ConnectWorldUpdateBegin(
        boost::bind( &controller_c::Update, this ) );

      _wb_fused_data = wb_fused_data_ptr( new wb_fused_data_c() );
      _wb_fused_data->load();

      std::stringstream state_log_name;
      state_log_name << "ppstate_" << trialid << ".log";
      state_log = log_ptr( new log_c( state_log_name.str() ) );
      if( !state_log || !state_log->open() ) {
        printf( "ERROR: unable to open ppstate log for writing\nPlugin failed to load\n" );
        return;
      }

      // -- FIN --
      printf( "controller has initialized\n" );
  
    }

    //-------------------------------------------------------------------------
    // Gazebo callback.  Called whenever the simulation advances a timestep
    virtual void Update( ) {

      static double vicon_latency = 0.01;

      static unsigned current_state_idx = 0;
      static bool first_state = true;
      static double sim_time = 0.0; 
      static double last_sim_time = 0.0;

      gazebo::math::Vector3 shell_pos, motor_pos;
      gazebo::math::Quaternion shell_rot, motor_rot;
      double motor_angle;

      // reference the current and next keyframe states
      wb_fused_state_ptr kf_state = _wb_fused_data->states[trialid].at( current_state_idx );
      wb_fused_state_ptr next_kf_state = _wb_fused_data->states[trialid].at( current_state_idx + 1 );

      double keyframe_time = ((double) current_state_idx) * vicon_latency;

      // if the current state has been integrated up to the sim time of the 
      // next state then increment the index.
      //if( _world->sim_time() >= next_kf_state->t() ) {
      if( _world->sim_time() >= keyframe_time ) {
        current_state_idx++;

        // if the index has reached the last state
        if( current_state_idx == _wb_fused_data->states[trialid].size() ) {
          // then kill the simulation
          state_log->close();
          exit(0);
        }

        // otherwise, update to the current and next keyframe states
        kf_state = _wb_fused_data->states[trialid].at( current_state_idx );
        next_kf_state = _wb_fused_data->states[trialid].at( current_state_idx+1 );

        // and pull state data from the keyframe
        shell_pos = gazebo::math::Vector3( kf_state->val(0), kf_state->val(1), kf_state->val(2) );
        shell_rot = gazebo::math::Quaternion( kf_state->val(6), kf_state->val(3), kf_state->val(4), kf_state->val(5) );

        motor_angle = kf_state->val(7);

        // set the pose for the shell
        gazebo::math::Pose pose( shell_pos, shell_rot );
        _weazelball->model()->SetLinkWorldPose( pose, _weazelball->shell() );

        // set the joint angle for the motor
        _weazelball->actuator()->SetPosition( 0, motor_angle );

        // then pull state of the motor from the simulator
        motor_pos = _weazelball->motor()->GetWorldCoGPose().pos;
        motor_rot = _weazelball->motor()->GetWorldCoGPose().rot;

      } else {
        // otherwise pull state data from the simulator
        shell_pos = _weazelball->shell()->GetWorldCoGPose().pos;
        shell_rot = _weazelball->shell()->GetWorldCoGPose().rot;
        motor_angle = _weazelball->actuator()->GetAngle( 0 ).Radian();
        motor_pos = _weazelball->motor()->GetWorldCoGPose().pos;
        motor_rot = _weazelball->motor()->GetWorldCoGPose().rot;
      }

      if( first_state ) {
        // update the sim time
        //sim_time = kf_state->t();
        sim_time = 0.0;
        _world->sim_time( sim_time );
      } else {
        sim_time = _world->sim_time();
        double dt = sim_time - last_sim_time;
      }

      //printf( "simtime[%f]\n", sim_time );

      last_sim_time = _world->sim_time();

      if( first_state ) first_state = false;
    }

    //-------------------------------------------------------------------------
    // Gazebo callback.  Called whenever the simulation is reset
    //virtual void Reset( ) { }

  };

  GZ_REGISTER_MODEL_PLUGIN( controller_c )

} // namespace gazebo

//-----------------------------------------------------------------------------

