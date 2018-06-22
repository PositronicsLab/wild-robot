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

//#include "log.h"
#include "models.h"
#include "weazelball.h"
#include "gazebo_log.h"

#include <stdlib.h>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>

//#include "movie.h"

#define PI 3.14159265359

//#define END_T 120.1f   
#define END_T 300.1f   

#define MAX_ITS 50

//#define MOTOR_FREQ 2.333
#define MOTOR_FREQ 2.5

#define SETTLING_TIME 0.250  // 250 ms

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

    //render_synchronization_buffer_c  rsync;

    //log_ptr log;                             //< state log
    wb_gazebo_session_ptr log;
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

      std::string world_name = model->GetWorld()->GetName();
      std::string engine_name = model->GetWorld()->GetPhysicsEngine()->GetType();
      printf("running %s using %s\n", world_name.c_str(), engine_name.c_str());
/*
      int r = rsync.open();
      if( r != 0 ) {
        printf("ERROR: rsync open return code %d\n", r );
      }
*/

      log = wb_gazebo_session_ptr( new wb_gazebo_session_c(wb_gazebo_session_c::WRITE) );
      if( log->open("sim.log") ) {
        printf( "ERROR: unable to open gazebo log for writing\nsimulation controller failed to load\n" );
        exit(1);
        return;
      }

      // -- FIN --
      printf( "randomization controller has initialized\n" ); 
    }

    //-------------------------------------------------------------------------
    // Gazebo callback.  Called whenever the simulation advances a timestep
    virtual void Update( ) {

      //rsync.synchronize( );
      double t = _world->sim_time();
      //if( t > END_T ) exit(0);

      static double t0, rtheta0;
      random_state(t, t0, rtheta0);

      //double dt = 0.001;

      //printf("t: %f, theta: %f\n", t, actuator->GetAngle( 0 ).Radian());

      double motor_freq = MOTOR_FREQ;
      double theta_0 = 0;
      //double theta_0 = rtheta0;
      double theta_t = actuator->GetAngle( 0 ).Radian();
      double omega = actuator->GetVelocity( 0 );

      double desired_omega = 2.0 * PI * motor_freq;
      double desired_theta_t = t * desired_omega + theta_0;
      //double desired_theta_t = t0 * desired_omega + theta_0;

      //printf("t: %f, theta_t:%f, desired_theta_t: %f\n", t, theta_t, desired_theta_t);

      double Kp = 0.3;
      double Kd = 0.03;

      double u = Kp * (desired_theta_t - theta_t) + Kd * (desired_omega - omega)
;
      if( u > 1e16 ) exit( 0 );

      //printf( "%f: %f\n", t, u );
      //double u = 0.001;

      actuator->SetForce( 0, u );

      //virtual_time += dt;

      log->write( t, _weazelball );
      //rsync.yield_to_render();
    }

    // NOTE: may need to return a value to indicate state seeded because the
    // sim needs to update before that state can be correctly read back from 
    // the simulator.  So, the calling function Update() may need to return
    // once before the state is processed if this function does update state
    bool random_state( double t, double& t0, double& theta0 ) {
      static bool first = true;
      static double ts = t;
      static double tf = ts + SETTLING_TIME;
      static int randomizations = 0;

      if( randomizations > MAX_ITS ) exit(0);
      if(t < tf && !first) return false;
      first = false;

      ts = t;
      tf = ts + SETTLING_TIME;
      t0 = ts;
      
      printf("ts: %f, tf: %f\n", ts, tf);
      // generate random state

      // position(x,y,0.041), orientation, linear_vel(dx,dy,0), omega, theta, 2pi*freq 
      // NOTE: a randomized actuator position will screw up controller if
      // the controller does not reset the theta_0 to the random position

      double x = (double)rand()/(double)RAND_MAX - 0.5;
      double y = (double)rand()/(double)RAND_MAX - 0.5;
      double z = 0.041;

      double qx = (double)rand()/(double)RAND_MAX - 0.5;
      double qy = (double)rand()/(double)RAND_MAX - 0.5;
      double qz = (double)rand()/(double)RAND_MAX - 0.5;
      double qw = (double)rand()/(double)RAND_MAX - 0.5;

      double Kd = 10.0;
      double dx = ((double)rand()/(double)RAND_MAX - 0.5) * Kd;
      double dy = ((double)rand()/(double)RAND_MAX - 0.5) * Kd;
      double dz = 0.0;

      double Komega = 10.0;
      double omegax = ((double)rand()/(double)RAND_MAX - 0.5) * Komega;
      double omegay = ((double)rand()/(double)RAND_MAX - 0.5) * Komega;
      double omegaz = ((double)rand()/(double)RAND_MAX - 0.5) * Komega;

      double Ktheta = 2.0;
      double theta = ((double)rand()/(double)RAND_MAX - 0.5) * Ktheta;
      double dtheta = 2.0 * PI * MOTOR_FREQ;

      gazebo::math::Vector3 pos = gazebo::math::Vector3(x,y,z);
      gazebo::math::Quaternion rot = gazebo::math::Quaternion(qw,qx,qy,qz);
      rot.Normalize();

      //gazebo::math::Vector3 lvel = gazebo::math::Vector3(dx,dy,dz);
      //gazebo::math::Vector3 avel = gazebo::math::Vector3(omegax,omegay,omegaz); 
      gazebo::math::Vector3 lvel = gazebo::math::Vector3(0,0,0);
      gazebo::math::Vector3 avel = gazebo::math::Vector3(0,0,0); 
      gazebo::math::Vector3 lacc = gazebo::math::Vector3(0,0,0);
      gazebo::math::Vector3 aacc = gazebo::math::Vector3(0,0,0);
      
      _weazelball->model()->Reset();
      _weazelball->model()->Update();

      _weazelball->model()->SetLinkWorldPose(gazebo::math::Pose(pos, rot), _weazelball->shell());
      //_weazelball->model()->SetLinearVel(lvel);
      //_weazelball->model()->SetAngularVel(avel);
      //_weazelball->model()->SetLinearAccel(lacc);
      //_weazelball->model()->SetAngularAccel(aacc);

      // if the joint angle is manipulated, energy is added to the system
      // best to not modify the joint at all in randomization
      //actuator->SetAngle(0, gazebo::math::Angle(theta));
      //actuator->SetPosition(0, theta);
      //actuator->SetVelocity(0, dtheta);
      //actuator->SetVelocity(0, 0);

      randomizations++;

      theta0 = theta;
      return true;
    }

  };

  GZ_REGISTER_MODEL_PLUGIN( controller_c )

} // namespace gazebo

//-----------------------------------------------------------------------------

