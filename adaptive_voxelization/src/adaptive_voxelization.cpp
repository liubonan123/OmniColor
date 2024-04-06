#include "ceres/ceres.h"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <pcl/ModelCoefficients.h>
#include <pcl/common/io.h>
#include <pcl/common/transforms.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/principal_curvatures.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <sstream>
#include <stdio.h>
#include <string>
#include <time.h>
#include <unordered_map>
#include "adaptive_voxelization.hpp"

using namespace std;
using namespace Eigen;

int main(int argc, char **argv)
{
     const string fileData = argv[1];
     // Calibration calib(fileData);
     std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> cloud;
     pcl::PointCloud<pcl::PointXYZRGB>::Ptr pc_base(new pcl::PointCloud<pcl::PointXYZRGB>);
     pcl::PointCloud<pcl::PointXYZI>::Ptr lidar_edge_clouds; 
     pcl::io::loadPCDFile(fileData, *pc_base);

     cout << "load original cloud" << endl;

     std::unordered_map<VOXEL_LOC, OctoTree *> adapt_voxel_map;
     double voxel_size;
     double eigen_threshold;
     voxel_size = 10;
     eigen_threshold = 0.0025;
     time_t t1 = clock();
     cout << "Adaptive Voxel building" << endl;
     for (size_t i = 0; i < pc_base->size();i++)
     {
          const pcl::PointXYZRGB& p_t = pc_base->points[i];
          Eigen::Vector3d pt(p_t.x, p_t.y, p_t.z);
          pcl::PointXYZI p_c;
          p_c.x = pt(0); p_c.y = pt(1); p_c.z = pt(2);
          float loc_xyz[3];
          for(int j = 0; j < 3; j++)
          {
               loc_xyz[j] = p_c.data[j] / voxel_size;
               if(loc_xyz[j] < 0) loc_xyz[j] -= 1.0;
               
          }
          // cout<<loc_xyz[0]<<loc_xyz[1]<<loc_xyz[2]<<endl;
          VOXEL_LOC position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]);
          auto iter = adapt_voxel_map.find(position);
          if(iter != adapt_voxel_map.end())
               adapt_voxel_map[position]->temp_points_.push_back(pt);
          else
          {
               OctoTree *octo_tree = new OctoTree(0, 1, eigen_threshold);
               adapt_voxel_map[position] = octo_tree;
               adapt_voxel_map[position]->quater_length_ = voxel_size / 4;
               adapt_voxel_map[position]->voxel_center_[0] = (0.5 + position.x) * voxel_size;
               adapt_voxel_map[position]->voxel_center_[1] = (0.5 + position.y) * voxel_size;
               adapt_voxel_map[position]->voxel_center_[2] = (0.5 + position.z) * voxel_size;
               cout<< (0.5 + position.x) * voxel_size<< (0.5 + position.y) * voxel_size<< (0.5 + position.z) * voxel_size<<endl;
               adapt_voxel_map[position]->temp_points_.push_back(pt);
               adapt_voxel_map[position]->new_points_++;
               Eigen::Vector3d layer_point_size(20, 20, 20);
               adapt_voxel_map[position]->layer_size_ = layer_point_size;
          }
     }
     for(auto iter = adapt_voxel_map.begin(); iter != adapt_voxel_map.end(); ++iter)
     {
          // down_sampling_voxel((iter->second->temp_points_), 0.0001);
          iter->second->init_octo_tree();
     }
     lidar_edge_clouds = pcl::PointCloud<pcl::PointXYZI>::Ptr(new pcl::PointCloud<pcl::PointXYZI>);
     pcl::PointCloud<pcl::PointXYZRGB> color_cloud_final;
     for (auto iter = adapt_voxel_map.begin(); iter != adapt_voxel_map.end(); ++iter)
     {
          std::vector<Plane*> plane_list;
          std::vector<Plane*> merge_plane_list;
          iter->second->get_plane_list(plane_list);

          if (plane_list.size() >= 1)
          {
               pcl::KdTreeFLANN<pcl::PointXYZI> kd_tree;
               pcl::PointCloud<pcl::PointXYZI> input_cloud;
               for(auto pv : iter->second->temp_points_)
               {
                    pcl::PointXYZI p;
                    p.x = pv[0]; p.y = pv[1]; p.z = pv[2];
                    input_cloud.push_back(p);
               }
               kd_tree.setInputCloud(input_cloud.makeShared());
               // std::cout << "origin plane size:" << plane_list.size() << std::endl;
               mergePlane(plane_list, merge_plane_list);
               for (auto plane : merge_plane_list)
               {
                    pcl::PointCloud<pcl::PointXYZRGB> color_cloud;
                    std::vector<unsigned int> colors;
                    colors.push_back(static_cast<unsigned int>(rand() % 256));
                    colors.push_back(static_cast<unsigned int>(rand() % 256));
                    colors.push_back(static_cast<unsigned int>(rand() % 256));
                    for(auto pv : plane->plane_points)
                    {
                         pcl::PointXYZRGB pi;
                         pi.x = pv[0]; pi.y = pv[1]; pi.z = pv[2];
                         pi.r = colors[0]; pi.g = colors[1]; pi.b = colors[2];
                         color_cloud.points.push_back(pi);
                         color_cloud_final.points.push_back(pi);
                    }
               }
               // std::cout << "merge plane size:" << merge_plane_list.size() << std::endl;
          }
     }
     cout << color_cloud_final.points.size() <<endl;
     time_t t2 = clock();
     pcl::io::savePCDFileBinary("color.pcd", color_cloud_final);
     down_sampling_voxel(*lidar_edge_clouds, 0.05);
     std::cout << "adaptive time:" << (double)(t2-t1)/(CLOCKS_PER_SEC) << "s" << std::endl;
}