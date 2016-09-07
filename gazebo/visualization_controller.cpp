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

//    log_ptr state_log;                       //< state log
  public:
    //-------------------------------------------------------------------------
    controller_c( void ) { }

    //-------------------------------------------------------------------------
    virtual ~controller_c( void ) {
//      //if( log ) log->close();

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
   
      // register update callback
      _updateConnection = event::Events::ConnectWorldUpdateBegin(
        boost::bind( &controller_c::Update, this ) );

      // load the motion capture data
      _wb_vicon_data = wb_vicon_data_ptr( new wb_vicon_data_c() );
      _wb_vicon_data->load_mocap();
/*
      // open the state log
      std::stringstream state_log_name;
      state_log_name << "trial_" << trialid << ".log";
      state_log = log_ptr( new log_c( state_log_name.str() ) );
      if( !state_log || !state_log->open() ) {
        printf( "ERROR: unable to open state log for writing\nPlugin failed to load\n" );
        return;
      }
*/
      // -- FIN --
      printf( "controller has initialized\n" );
  
    }

    //-------------------------------------------------------------------------
    // Gazebo callback.  Called whenever the simulation advances a timestep
    virtual void Update( ) {

      static std::vector< std::vector<double> > vicon_data_stats(10);
 
      static unsigned calls = 0;
      static unsigned current_state_idx = 0;

      static bool first_state = true;

      static double virtual_time = 0.0; 
      static double last_virtual_time = 0.0;
      static double adjusted_vicon_time = 0.0; 
      static double initial_vicon_time = 0.0;

      static unsigned motor_cycles = 0;
      static double last_theta = 0.0;

      gazebo::math::Vector3 pos;
      gazebo::math::Quaternion rot;

      if( calls == 0 ) {
        if( current_state_idx == _wb_vicon_data->states[trialid].size() ) {
          // kills the simulation
//          state_log->close();

          exit(0);
        }

        wb_vicon_state_ptr state = _wb_vicon_data->states[trialid].at( current_state_idx++ );

        pos = gazebo::math::Vector3( state->val(0), state->val(1), state->val(2) );
        rot = gazebo::math::Quaternion( state->val(6), state->val(3), state->val(4), state->val(5) );
        // the vicon data is recorded with a model frame rotated by 90 degrees 
        // to the world frame.  Rotate the model frame back to a frame aligned
        // with the world.
        rot = rot * gazebo::math::Quaternion( -PI/2.0, 0.0, 0.0 );

        gazebo::math::Pose pose( pos, rot );

        _weazelball->model()->SetLinkWorldPose( pose, _weazelball->shell() );

        if( !first_state ) {
          double dt = 1.0 / vicon_sample_rate;
          virtual_time += dt;
        } else {
          initial_vicon_time = state->t();
        }
        adjusted_vicon_time = state->t() - initial_vicon_time;

        _world->sim_time( virtual_time );
        //printf( "vicon_time[%f], virtual_time[%f]\n", vicon_time, virtual_time );

        //last_vicon_time = state->t();
        last_virtual_time = virtual_time;
      } else {
        //if( calls >= 25 ) {
        if( calls >= 1 ) {
          calls = 0;
          return;
        }
      }
/*
      double theta;
      double motor_freq = WEAZELBALL_MOTOR_HZ;
      double theta_at_sim_t_zero;

      // updating the internal state of the robot
      if( trialid == 0 ) {
        //theta_at_sim_t_zero = 2.8762;
        theta_at_sim_t_zero = 3.9623;
      } else if( trialid == 1 ) {
        theta_at_sim_t_zero = 3.5203;
      } else if( trialid == 2 ) {
        theta_at_sim_t_zero = 1.7233;
      } else if( trialid == 3 ) {
        theta_at_sim_t_zero = 5.3984;
      } else if( trialid == 4 ) {
        theta_at_sim_t_zero = 1.5870;
      } else if( trialid == 5 ) {
        theta_at_sim_t_zero = 4.9115;
      } else if( trialid == 6 ) {
        theta_at_sim_t_zero = 2.1809;
      } else if( trialid == 7 ) {
        theta_at_sim_t_zero = 4.6411;
      } else if( trialid == 8 ) {
        theta_at_sim_t_zero = 5.4913;
      } else if( trialid == 9 ) {
        theta_at_sim_t_zero = 1.7763;
      }
      double omega = 2.0 * PI * motor_freq;  // radians per sec

      theta = virtual_time * omega + theta_at_sim_t_zero;
      //theta -= ((double)((int)( theta / (2.0 * PI) )) * (2.0 * PI));
      while( theta - 2.0 * PI > 0.0 )
        theta -= 2.0 * PI;

      if( !first_state ) {
        if( last_theta + PI / 10.0 > 2.0 * PI && theta - PI / 10.0 < 0.0 )
          motor_cycles++;
      }

      last_theta = theta;
      //}
*/

      printf( "virtual_time[%f]", virtual_time );
      printf( ", vicon_time[%f]", adjusted_vicon_time );
//      printf( ", cycles[%u]", motor_cycles );
//      printf( ", theta[%f]", theta );
      printf( "\n" );


      if( first_state ) first_state = false;
/*
      if( calls == 0 ) {
        // log state

        std::stringstream data;
        data << virtual_time << " ";
        data << adjusted_vicon_time << " ";
        data << pos.x << " " << pos.y << " " << pos.z << " ";
        data << rot.x << " " << rot.y << " " << rot.z << " " << rot.w << " ";
//        data << theta << std::endl;       

        state_log->write( data.str() ); 
      }
*/
      calls++;
    }

  };

  GZ_REGISTER_MODEL_PLUGIN( controller_c )

} // namespace gazebo

//-----------------------------------------------------------------------------

