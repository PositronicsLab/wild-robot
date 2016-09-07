/*
This controller is designed to replay the raw vicon data so as visualize the
state of the Weazelball throughout the experiments
*/
#include <gazebo/gazebo.hh>
#include <gazebo/common/Plugin.hh>
#include <gazebo/common/common.hh>
#include <gazebo/common/Events.hh>
#include <gazebo/physics/physics.hh>

#include "common.h"
#include "log.h"
#include "models.h"
#include "vicon.h"

//-----------------------------------------------------------------------------
// set on the interval [1,10] to switch between different vicon motion capture
// sessions.
//unsigned trial_number = 10;

// the zero index based trial index used when accessing the container data
//unsigned trial_idx = EXPERIMENTAL_SESSION_ID - 1;

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

    world_ptr _world;                        //< wraps Gazebo::WorldPtr
    weazelball_ptr _weazelball;              //< wraps Gazebo::ModelPtr

//    wb_vicon_data_ptr _wb_vicon_data;        //< Vicon data container
    wb_vicon_session_ptr _vicon_session;

    log_ptr log;

  public:
    //-------------------------------------------------------------------------
    /// Default Constructor
    controller_c( void ) { }

    //-------------------------------------------------------------------------
    /// Destructor
    virtual ~controller_c( void ) {
      if( log ) log->close();

      // unregister the update callback
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

      // reset the world to begin
      _world->reset();

      // get references to objects in the world
      _weazelball = _world->weazelball();
 
      // register update callback
      _updateConnection = event::Events::ConnectWorldUpdateBegin(
        boost::bind( &controller_c::Update, this ) );

      // load the motion capture data
      //_wb_vicon_data = wb_vicon_data_ptr( new wb_vicon_data_c() );
      //_wb_vicon_data->load_mocap();
      //printf( "Loaded %u sessions\n", _wb_vicon_data->sessions.size() );

      _vicon_session = wb_vicon_session_c::load( EXPERIMENTAL_SESSION_ID );

      log = log_ptr( new log_c( "height.log" ) );
      if( !log || !log->open() ) {
        printf( "ERROR: unable to open height log\nPlugin failed to load\n" );
        return;
      }

      // -- FIN --
      printf( "controller has initialized\n" );
  
    }

    //-------------------------------------------------------------------------
    // Gazebo callback.  Called whenever the simulation advances a timestep
    virtual void Update( ) {

      static unsigned calls = 0;               // tracks calls to this function
      static unsigned current_state_idx = 0;   // index of the current state

      static bool first_state = true;          // indicates if first state

      static double virtual_time = 0.0;        // virtual time of the sim
      static double adjusted_vicon_time = 0.0; // normalized vicon time 
      static double initial_vicon_time = 0.0;  // to normalize vicon time

      // check if the system has reached the last state
      //if( current_state_idx == _wb_vicon_data->sessions[trial_idx]->states.size() ) {
      if( current_state_idx == _vicon_session->states.size() ) {
        // if so then kill the simulation
        exit(0);
      }

      // otherwise, read the current state from the vicon data
      //wb_vicon_state_ptr state = _wb_vicon_data->sessions[trial_idx]->states[current_state_idx++];
      wb_vicon_state_ptr state = _vicon_session->states[current_state_idx++];

      // then set the pose in the simulation
      _weazelball->pose( state );

      // check if this is the first update 
      if( first_state ) {
        // if so then set the initial time from the state
        initial_vicon_time = state->t();
      } else {
        // otherwise, compute the ideal time-step and update virtual time
        double dt = 1.0 / VICON_SAMPLE_RATE;
        virtual_time += dt;
      }

      // adjust the vicon time by taking the difference between vicon time for
      // current vicon state and the initial vicon state
      adjusted_vicon_time = state->t() - initial_vicon_time;

      // update the world virtual time
      _world->sim_time( virtual_time );

      // log time data
      printf( "virtual_time[%f]", virtual_time );
      printf( ", vicon_time[%f]", adjusted_vicon_time );
      printf( "\n" );

      std::stringstream ss;
      ss << virtual_time << " ";
      ss << state->val(0) << " ";  // x
      ss << state->val(1) << " ";  // y
      ss << state->val(2) << " ";  // z
      ss << state->val(3) << " ";
      ss << state->val(4) << " ";
      ss << state->val(5) << " ";
      ss << state->val(6) << std::endl;

//      math::Quaternion q( state->val(6), state->val(3), state->val(4), state->val(5) );
//      math::Vector3 zp = q.RotateVector( math::Vector3(0,-1,0) );
//      ss << zp.x << " ";
//      ss << zp.y << " ";
//      ss << zp.z << std::endl;

      log->write( ss.str() );

      // disable first_state flag
      if( first_state ) first_state = false;

      // increment the number of calls to update
      calls++;
    }

  };

  GZ_REGISTER_MODEL_PLUGIN( controller_c )

} // namespace gazebo

//-----------------------------------------------------------------------------

