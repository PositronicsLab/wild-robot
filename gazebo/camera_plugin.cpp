#include <gazebo/math/Rand.hh>
#include <gazebo/gui/GuiIface.hh>
#include <gazebo/rendering/rendering.hh>
#include <gazebo/gazebo.hh>

#include <sstream>

#include "log.h"
#include "movie.h"

#define SAVE_FRAMES 1
#define TRACKING_CAM 0

#define SYNCHRONIZED 1


namespace gazebo
{
  class render_plugin_c : public SystemPlugin
  {
  public:
    render_plugin_c( void ) { }

    virtual ~render_plugin_c( void ) {
      this->connections.clear();
      if (this->userCam) {
        this->userCam->EnableSaveFrame(false);
      }
      this->userCam.reset();
/*
      if( time_log )
        time_log->close(); 
*/
      if(SYNCHRONIZED) {
        close( _shmfd );
      }
    }

    void Load(int /*argc*/, char ** /*argv*/) {
      this->connections.push_back(
          event::Events::ConnectPreRender(
            boost::bind(&render_plugin_c::Update, this)));
      _camdata = NULL;

      if(SYNCHRONIZED) {
        _shmfd = shm_open( RENDER_SYNCH_CHANNEL, O_RDWR, S_IRUSR | S_IWUSR );
        if( _shmfd == -1 ) {
          std::cerr << "ERROR(camera_plugin.cpp):shm_open()" << std::endl;
          return;
        }
        if( ftruncate( _shmfd, sizeof(camdata_t) ) == -1 ) {
          std::cerr << "ftruncate()" << std::endl;
          return;
        }
        void* addr;
        addr = mmap( NULL, sizeof(camdata_t), PROT_READ | PROT_WRITE, MAP_SHARED, _shmfd, 0 );
        if( addr == MAP_FAILED ) {
          std::cerr << "mmap( _camdata )" << std::endl;
          close( _shmfd );
          return;
        }
        _camdata = static_cast<camdata_t*> ( addr );
      }

      //frameduration = (double)1.0f / (double)OUTPUT_FPS;
      std::cout << "camera_plugin loaded\n";
    }

    void Init( void ) {

    }

    void Update( void ) {
      //std::cout << "Update()" << std::endl;
      // Get scene pointer
      rendering::ScenePtr scene = rendering::get_scene();

      // Wait until the scene is initialized.
      if( ( !scene || !scene->GetInitialized() ) ) {
        return;
      }

      //gazebo::common::Color grey( 0.7, 0.7, 0.7, 1.0 );
      //scene->SetBackgroundColor( grey );
      //scene->SetAmbientColor( grey );

      static int frame = 1;
      static gazebo::common::Time last_sim_time = scene->GetSimTime();
      gazebo::common::Time sim_time = scene->GetSimTime();

      if( !this->userCam ) {
        // Get a pointer to the active user camera
        this->userCam = gui::get_active_camera();
        // Specify the path to save frames into

        if( SAVE_FRAMES ) {
          std::stringstream ss_path;
          static std::string scene_name = scene->GetName();
  
          ss_path << "/tmp/gazebo_frames/" << scene_name;
/*
          if( !time_log ) {
            std::stringstream ss_log;
            ss_log << ss_path.str() << "/time.log";
            time_log = boost::shared_ptr<log_c>( new log_c( ss_log.str() ) );
            time_log->open();
  
            std::string log_hdr = "frame,sim_time\n";
            time_log->write( log_hdr );
          }
*/
          this->userCam->SetSaveFramePathname( ss_path.str() );
          // Enable saving frames for initial frame
          this->userCam->EnableSaveFrame(true);
          //prevframe = 0.0f;
          //nextframe = prevframe + frameduration;
        }
      } else {
        if(SYNCHRONIZED) {
          bool waiting = false;
          pthread_mutex_lock( &_camdata->mutex );
          if( _camdata->state == 0 ) {
            waiting = true;
          }
          if( waiting ) {
            // issue has to be in here somewhere
            this->userCam->EnableSaveFrame(false);
            _camdata->rendering = 0;
            pthread_mutex_unlock( &_camdata->mutex );
            return;
          }
          pthread_mutex_unlock( &_camdata->mutex );
        }
      }

      if( TRACKING_CAM ) {
        if( _camdata ) {
          bool updated = false;
          while( !updated ) {
             pthread_mutex_lock( &_camdata->mutex );
             if( _camdata->state == 1 ) updated = true;
             pthread_mutex_unlock( &_camdata->mutex );
          }

          gazebo::math::Pose p;

          rendering::LightPtr     light;
          light = scene->GetLight( "light01" );

          pthread_mutex_lock( &_camdata->mutex );
          std::cout << _camdata->pose << std::endl;
          // compute light pose
          if( light ) {
            p = _camdata->pose;
            p.pos += p.rot.RotateVector( gazebo::math::Vector3(0,0,10) );
            //p.rot *= gazebo::math::Quaternion(0, 1.5708, 1.5708);
            //_light->ShowVisual( false );
            light->SetPosition( p.pos );
            light->SetRotation( p.rot );
          }
          // compute camera pose
          p = _camdata->pose;
          p.pos += p.rot.RotateVector( gazebo::math::Vector3(0,0.08,0.2) );
          p.rot *= gazebo::math::Quaternion(0, 1.5708, 1.5708);
          this->userCam->SetWorldPose( p );
          //this->userCam->SetWorldPose( _camdata->pose );
          _camdata->state = 0;
          pthread_mutex_unlock( &_camdata->mutex );
        }
      } else {
        if(SYNCHRONIZED) {
          if( _camdata ) {
            bool updated = false;
            while( !updated ) {
               pthread_mutex_lock( &_camdata->mutex );
               if( _camdata->state == 1 ) {
                 //printf("s:1\n");
                 updated = true;
                 _camdata->state = 0;
                 //printf("s:0\n");
               }
               pthread_mutex_unlock( &_camdata->mutex );
            }
            //while( _camdata->state != 1 );
          }
        }
      }

      if(SAVE_FRAMES) {
        // Enable saving frames
        //if( sim_time.Double() >= nextframe ) {
          pthread_mutex_lock( &_camdata->mutex );
          _camdata->rendering = 1;
          pthread_mutex_unlock( &_camdata->mutex );
          //prevframe = nextframe;
          //nextframe = prevframe + frameduration;
          this->userCam->EnableSaveFrame(true);
/*
          std::stringstream ss_log_data;
          ss_log_data << frame << "," << sim_time.Double() << std::endl;
          time_log->write( ss_log_data.str() );
*/
        //}
      }

      last_sim_time = sim_time;
      frame++;
    }

    /// Pointer the user camera.
    private: rendering::UserCameraPtr userCam;

    /// All the event connections.
    private: std::vector<event::ConnectionPtr> connections;

    private: boost::shared_ptr<log_c> time_log;

    private: rendering::VisualPtr visBox;

    private: int          _shmfd;
    private: camdata_t*   _camdata;

  };

  // Register this plugin with the simulator
  GZ_REGISTER_SYSTEM_PLUGIN( render_plugin_c )
}

