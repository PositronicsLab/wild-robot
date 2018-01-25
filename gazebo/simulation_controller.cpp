/*
This controller was cloned from the vicon visualization controller after 
refactoring.  It still contains logging code which was culled from the vicon
controller.  This controller can play the basis of another controller that
processes the data rather than modifying the vicon controller further.  This is
to ensure that a controller exists for each step in post-processing
*/
#include <gazebo/gazebo.hh>
#include <gazebo/common/Plugin.hh>
#include <gazebo/common/common.hh>
#include <gazebo/common/Events.hh>
#include <gazebo/physics/physics.hh>

#include "log.h"
#include "models.h"
#include "weazelball.h"

#include <stdlib.h>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>

#define PI 3.14159265359

//-----------------------------------------------------------------------------
// set on the interval [0,9] to switch between different vicon motion capture
// sessions.
unsigned trialid = 9;

double vicon_sample_rate = 100.0;      // set in software for these experiments

//-----------------------------------------------------------------------------
/**
The gazebo plugin visualization controller
**/
namespace gazebo 
{
  class controller_c : public ModelPlugin
  {
  private:
    event::ConnectionPtr _updateConnection;  //< Gazebo update callback

    world_ptr _world;                        //< Gazebo world pointer
    weazelball_ptr _weazelball;              //< Weazelball data structure

    wb_vicon_data_ptr _wb_vicon_data;        //< Vicon data structure

    physics::JointPtr actuator;
    double virtual_time;

//    log_ptr state_log;                       //< state log
  public:
    //-------------------------------------------------------------------------
    controller_c( void ) { }

    //-------------------------------------------------------------------------
    virtual ~controller_c( void ) {

      event::Events::DisconnectWorldUpdateBegin( _updateConnection );
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

      _world->reset();

      // get references to objects in the world
      _weazelball = _world->weazelball();
   
      // register update callback
      _updateConnection = event::Events::ConnectWorldUpdateBegin(
        boost::bind( &controller_c::Update, this ) );

      actuator = _weazelball->actuator();
      virtual_time = 0.0;   

      // -- FIN --
      printf( "controller has initialized\n" );
  
    }

    //-------------------------------------------------------------------------
    // Gazebo callback.  Called whenever the simulation advances a timestep
    virtual void Update( ) {
      double t = _world->sim_time();
      double dt = 0.001;

      double motor_freq = 2.333;
      double theta_0 = 0;
      double theta_t = actuator->GetAngle( 0 ).Radian();
      double omega = actuator->GetVelocity( 0 );

      double desired_omega = 2.0 * PI * motor_freq;
      double desired_theta_t = t * desired_omega + theta_0;

      double Kp = 0.3;
      double Kd = 0.03;

      double u = Kp * (desired_theta_t - theta_t) + Kd * (desired_omega - omega)
;
      if( u > 1e16 ) exit( 0 );

      printf( "%f: %f\n", t, u );
      //double u = 0.001;

      actuator->SetForce( 0, u );

      virtual_time += dt;
    }
  };

  GZ_REGISTER_MODEL_PLUGIN( controller_c )

} // namespace gazebo

//-----------------------------------------------------------------------------
