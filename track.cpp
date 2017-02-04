
#include "OpenNI.h"
#include "NiteSampleUtilities.h"
#include <iostream>
#include <NiTE.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


#include "SDL/SDL.h"
#include "SDL/SDL_opengl.h"
#include "SDL/SDL_image.h"
#include "string.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

using namespace std;
using namespace cv;
using namespace openni;

float x_skeleton = 0, y_skeleton = 0;

cv::Scalar colors[] = 
{
	cv::Scalar( 0xFF, 0, 0 ),
	cv::Scalar( 0, 0xFF, 0 ),
	cv::Scalar( 0, 0, 0xFF ),
	cv::Scalar( 0xFF, 0xFF, 0 ),
	cv::Scalar( 0xFF, 0, 0xFF ),
	cv::Scalar( 0, 0xFF, 0xFF ),
};


cv::Mat depthToImage( nite::UserTrackerFrameRef& userFrame )
{
	cv::Mat depthImage;
	openni::VideoFrameRef depthFrame = userFrame.getDepthFrame();
	if ( depthFrame.isValid()) 
	{
		openni::VideoMode videoMode = depthFrame.getVideoMode();
		depthImage = cv::Mat( videoMode.getResolutionY(),videoMode.getResolutionX(),CV_16SC1,(short*) depthFrame.getData() );

		depthImage.convertTo(depthImage, CV_8UC1, 255.0/10000);
		cv::cvtColor(depthImage, depthImage, CV_GRAY2BGR);
	}
	return depthImage;
}

void drawUser( nite::UserTrackerFrameRef& userFrame, cv::Mat& image )
{
	openni::VideoFrameRef depthFrame = userFrame.getDepthFrame();
	if ( depthFrame.isValid() ) 
	{
		openni::VideoMode videoMode = depthFrame.getVideoMode();
		cv::Mat pMapLabel = cv::Mat(videoMode.getResolutionY(),videoMode.getResolutionX(),CV_16SC1,(short*) userFrame.getUserMap().getPixels());
		pMapLabel.convertTo(pMapLabel,CV_8UC1);

		for(int i = 0; i < 6; i++)
		{
			cv::Mat mask;
			cv::compare(pMapLabel, i+1, mask, CV_CMP_EQ);
			cv::add(image, colors[i], image, mask);
		}
	}
}

int drawSkeleton( nite::UserTrackerFrameRef& userFrame, nite::UserTracker& userTracker, cv::Mat& image)
{
	const nite::Array<nite::UserData>& users = userFrame.getUsers();
	for ( int i = 0; i < users.getSize(); ++i ) 
	{
		const nite::UserData& user = users[i];
		if ( user.isNew() ) 
		{
			userTracker.startSkeletonTracking( user.getId() );
			userTracker.startPoseDetection( user.getId(), nite::POSE_PSI);
			userTracker.startPoseDetection( user.getId(), nite::POSE_CROSSED_HANDS);
		}
		
		else if ( !user.isLost() ) 
		{
			if (user.getSkeleton().getState() == nite::SKELETON_TRACKED) 
			{
				for ( int j = 6; j < 7; j++ ) 
				{
					//joint = user.getSkeleton().getJoint((nite::JointType)j);
					if(user.getSkeleton().getJoint((nite::JointType)j).getPositionConfidence() >= 0.7f) 
					{
						//position = user.getSkeleton().getJoint((nite::JointType)j).getPosition();
						
						userTracker.convertJointCoordinatesToDepth(user.getSkeleton().getJoint((nite::JointType)j).getPosition().x, user.getSkeleton().getJoint((nite::JointType)j).getPosition().y, user.getSkeleton().getJoint((nite::JointType)j).getPosition().z, &x_skeleton, &y_skeleton );
						cv::circle(image, cvPoint((int)x_skeleton,(int)y_skeleton),1, cv::Scalar( 0, 0, 255 ), 5);
					}
				}
			}
		}
	}
	//cout << x_skeleton << endl;
	return x_skeleton;
}



int main(int argc, char **argv)
{

	float Xcor;
	int x, z;
	char host_ip[] = "127.0.0.1";
	short host_port = 1236;
	char remote_ip[] = "127.0.0.1";
	short remote_port = 7896;
	int i;
	int host_sock, remote_sock;
	char datagram[512];
	int remote_len;

	struct sockaddr_in host_add;
	struct sockaddr_in remote_add;

	host_sock = socket(AF_INET, SOCK_DGRAM, 0);
	
	host_add.sin_family = AF_INET;
	host_add.sin_port = htons(host_port);
	host_add.sin_addr.s_addr = inet_addr(host_ip);

	z = bind(host_sock, (struct sockaddr *)&host_add, sizeof(host_add));

	nite::UserTracker userTracker;
	nite::Status niteRc;
	nite::NiTE::initialize();
	niteRc = userTracker.create();
	nite::UserTrackerFrameRef userTrackerFrame;



	openni::Device device; 
	openni::VideoStream color;
	openni::VideoFrameRef colorFrame; 
	openni::VideoStream depth;
	openni::VideoFrameRef depthFrame;
	openni::Status rc;
	
	cv::Mat colourImage;
	cv::Mat depthImage;
	
	rc = openni::OpenNI::initialize();
	if (rc != openni::STATUS_OK)
	{
		printf("Failed to initialize OpenNI\n%s\n", openni::OpenNI::getExtendedError());
		return rc;
	}

	rc = device.open(openni::ANY_DEVICE);
	rc = color.create(device, openni::SENSOR_COLOR);
	rc = color.start();
	rc = depth.create(device, openni::SENSOR_DEPTH);
	rc = depth.start();


    //MAIN GAME LOOP :


	while(1)
	{


		color.readFrame(&colorFrame);

		const openni::RGB888Pixel* imageBuffer = (const openni::RGB888Pixel*)colorFrame.getData();
		
		colourImage.create(colorFrame.getHeight(), colorFrame.getWidth(), CV_8UC3);
		memcpy( colourImage.data, imageBuffer, 3*colorFrame.getHeight()*colorFrame.getWidth()*sizeof(uint8_t) );
		cvtColor(colourImage, colourImage, CV_BGR2RGB); 

		niteRc = userTracker.readFrame(&userTrackerFrame);
		if (niteRc != nite::STATUS_OK)
		{
			//printf("Get next frame failed\n");
			continue;
		}
		depthImage = depthToImage(userTrackerFrame);
		drawUser(userTrackerFrame, depthImage);

		bzero(datagram, 512); 

		remote_sock = socket(AF_INET, SOCK_DGRAM, 0);
		remote_add.sin_family = AF_INET;
		remote_add.sin_port = htons(remote_port);
		remote_add.sin_addr.s_addr = inet_addr(remote_ip);


		//GAME LOGIC :
		Xcor = drawSkeleton(userTrackerFrame, userTracker, depthImage) * 5;
        //cout << Xcor << endl;
		sprintf(datagram, "%f", Xcor);
		x = sendto(remote_sock, datagram, strlen(datagram), 0, (struct sockaddr *)&remote_add, sizeof(remote_add));
		cout << datagram << endl;        

        

		cv::imshow("depth image", depthImage);

		int key = cv::waitKey( 10 );
		if ( key == 'q' || key == 0x1b ) 
		{
			break;
		}
		close(remote_sock);
		
	}
	close(host_sock);
	nite::NiTE::shutdown();

    //allows user to exit

	return 0;
}