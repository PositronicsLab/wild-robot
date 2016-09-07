/*
This controller is designed to replay the raw vicon data so as visualize the
state of the Weazelball throughout the experiments
*/
#include <gazebo/gazebo.hh>
#include <gazebo/common/Plugin.hh>
#include <gazebo/common/common.hh>
#include <gazebo/common/Events.hh>
#include <gazebo/physics/physics.hh>

#define FILTER

#include "common.h"
#include "log.h"
#include "models.h"
#include "vicon.h"
#include "video.h"
#include "math.h"
#include "weazelball.h"

#include <unistd.h>
//-----------------------------------------------------------------------------
// set on the interval [1,10] to switch between different vicon motion capture
// sessions.
//unsigned trial_number = 1;

// the zero index based trial index used when accessing the container data
//unsigned trial_idx = trial_number - 1;

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
    led_ptr _led;

    //wb_vicon_data_ptr _wb_vicon_data;        //< Vicon data container
    wb_vicon_session_ptr _vicon_session;
    wb_video_session_ptr _wb_video_session;
    wb_video_event_ptr _wb_event;

    wb_session_ptr _wb_session;

  public:
    //-------------------------------------------------------------------------
    /// Default Constructor
    controller_c( void ) { }

    //-------------------------------------------------------------------------
    /// Destructor
    virtual ~controller_c( void ) {
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
      _led = _world->led(); 
      _led->switch_off();
 
      // register update callback
      _updateConnection = event::Events::ConnectWorldUpdateBegin(
        boost::bind( &controller_c::Update, this ) );

      // load the motion capture data
      //_wb_vicon_data = wb_vicon_data_ptr( new wb_vicon_data_c() );
      //_wb_vicon_data->load_mocap();
      //printf( "Loaded %u sessions\n", _wb_vicon_data->sessions.size() );

      _vicon_session = wb_vicon_session_c::load( EXPERIMENTAL_SESSION_ID );

      _wb_session = wb_session_ptr( new wb_session_c( ) );
      _wb_session->evaluate( _vicon_session );
      _wb_session->interpolate();

      // load the video data
      _wb_video_session = wb_video_session_ptr( new wb_video_session_c() );
      _wb_video_session->read( EXPERIMENTAL_SESSION_ID );
      printf( "video_t0[%d], vicon_t0[%f]\n", _wb_video_session->base_safe_start_frame(), _wb_video_session->vicon_start_time() );
      assert( _wb_video_session->events.size() );
      _wb_event = _wb_video_session->events[0];

      for( unsigned i = 0; i < _wb_video_session->events.size(); i++ ) {
        wb_video_event_ptr e = _wb_video_session->events[i];
        if( !e->is_gap() ) {
          // prints the video events
          //printf( "event[%u]: frame_id[%d], start[%f], end[%f]\n", e->index(), e->frame_id(), e->virtual_start_time(), e->virtual_end_time() );
        }
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

      static double vicon_step = 1.0/ _wb_video_session->vicon_sample_rate();
      wb_vicon_state_ptr state;

      static unsigned initial_vicon_state = _wb_video_session->vicon_states_to_ignore();

      // check if the system has reached the last state
      //if( current_state_idx == _wb_vicon_data->sessions[trial_idx]->states.size() ) {
      //if( current_state_idx == _vicon_session->states.size() ) {
      if( current_state_idx == _wb_session->shell_states.size() ) {
        // if so then kill the simulation
        exit(0);
      }

      // otherwise, read the current state from the vicon data
      //wb_vicon_state_ptr state = _wb_vicon_data->sessions[trial_idx]->states[current_state_idx++];
      if( first_state ) {
        current_state_idx = initial_vicon_state;
      }
      state = _vicon_session->states[current_state_idx];

      // then set the pose in the simulation
      _weazelball->pose( state );

      // check if this is the first update 
      if( first_state ) {
        // if so then set the initial time from the state
        initial_vicon_time = state->t();
      } else {
        // otherwise, compute the ideal time-step and update virtual time
        //double dt = 1.0 / VICON_SAMPLE_RATE;
        virtual_time += vicon_step;
      }

      // adjust the vicon time by taking the difference between vicon time for
      // current vicon state and the initial vicon state
      adjusted_vicon_time = state->t() - initial_vicon_time;

      // update the world virtual time
      _world->sim_time( virtual_time );

      assert( _wb_event );
      if( !_wb_event->is_gap() ) {
        if( virtual_time >= _wb_event->virtual_start_time() ) {
          if( virtual_time <= _wb_event->virtual_end_time() ) {
            _led->switch_on();
          } else {
            _led->switch_off();
            unsigned event_id = _wb_event->index();
            if( event_id < _wb_video_session->events.size() - 1 )  {
              _wb_event = _wb_video_session->events[event_id + 1];
              if( _wb_event->is_gap() ) {
                if( event_id + 1 < _wb_video_session->events.size() - 1 )  {
                  _wb_event = _wb_video_session->events[event_id + 2];
                }
              }
            }
          }
        }
      }

      // log time data
      printf( "%d :", current_state_idx );
      printf( ", virtual_time[%f]", virtual_time );
      printf( ", vicon_stream_ts[%f]", adjusted_vicon_time );
      printf( ", frame[%u]", _wb_video_session->frame( virtual_time ) );
      printf( ", vicon_idx[%u]", current_state_idx );
      printf( "\n" );


      // disable first_state flag
      if( first_state ) first_state = false;

      // increment the number of calls to update
      calls++;
      current_state_idx++;

      usleep( 10000 );
    }

  };

  GZ_REGISTER_MODEL_PLUGIN( controller_c )

} // namespace gazebo

//-----------------------------------------------------------------------------

