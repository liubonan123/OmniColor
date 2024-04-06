#ifndef ADAPTIVE_VOXELIZATION_HPP
#define ADAPTIVE_VOXELIZATION_HPP

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


#define max_layer 8

typedef struct Plane
{
  pcl::PointXYZINormal p_center;
  Eigen::Vector3d center;
  Eigen::Vector3d normal;
  Eigen::Matrix3d covariance;
  std::vector<Eigen::Vector3d> plane_points;
  float radius = 0;
  float min_eigen_value = 1;
  float d = 0;
  int points_size = 0;
  bool is_plane = false;
  bool is_init = false;
  int id;
  bool is_update = false;
} Plane;

class OctoTree
{
public:
  std::vector<Eigen::Vector3d> temp_points_;
  Plane* plane_ptr_;
  int layer_;
  int octo_state_; // 0 is end of tree, 1 is not
  OctoTree* leaves_[8];
  double voxel_center_[3]; // x, y, z
  Eigen::Vector3d layer_size_;
  float quater_length_;
  float planer_threshold_;
  int points_size_threshold_;
  int update_size_threshold_;
  int new_points_;
  bool init_octo_;
  bool update_enable_;

  OctoTree(int layer, int points_size_threshold, float planer_threshold):
    layer_(layer), points_size_threshold_(points_size_threshold),
    planer_threshold_(planer_threshold)
  {
    temp_points_.clear();
    octo_state_ = 0;
    new_points_ = 0;
    update_size_threshold_ = 5;
    init_octo_ = false;
    update_enable_ = true;
    for(int i = 0; i < 8; i++)
      leaves_[i] = nullptr;
    plane_ptr_ = new Plane;
  }

  void init_plane(const std::vector<Eigen::Vector3d>& points, Plane* plane)
  {
    plane->covariance = Eigen::Matrix3d::Zero();
    plane->center = Eigen::Vector3d::Zero();
    plane->normal = Eigen::Vector3d::Zero();
    plane->points_size = points.size();
    plane->radius = 0;
    for(auto pv : points)
    {
      plane->covariance += pv * pv.transpose();
      plane->center += pv;
    }
    plane->center = plane->center / plane->points_size;
    plane->covariance = plane->covariance / plane->points_size -
                        plane->center * plane->center.transpose();                     
    // cout<<"points_size"<<endl;
    // cout<< plane->points_size<<endl;
    // cout << "center" << endl;               
    // cout<< plane->center <<endl;
    // cout << "cova" << endl;
    // cout<< plane->covariance <<endl;
    Eigen::EigenSolver<Eigen::Matrix3d> es(plane->covariance);
    Eigen::Matrix3cd evecs = es.eigenvectors();
    Eigen::Vector3cd evals = es.eigenvalues();
    Eigen::Vector3d evalsReal;
    // cout<< "evalsReal" <<endl;
    evalsReal = evals.real();
    Eigen::Matrix3f::Index evalsMin, evalsMax;
    evalsReal.rowwise().sum().minCoeff(&evalsMin);
    evalsReal.rowwise().sum().maxCoeff(&evalsMax);
    int evalsMid = 3 - evalsMin - evalsMax;
    // cout<< evalsReal(evalsMax) << evalsReal(evalsMid) << evalsReal(evalsMin) <<endl;
    Eigen::Vector3d evecMin = evecs.real().col(evalsMin);
    Eigen::Vector3d evecMid = evecs.real().col(evalsMid);
    Eigen::Vector3d evecMax = evecs.real().col(evalsMax);

    if(evalsReal(evalsMin) < planer_threshold_ && evalsReal(evalsMid) > 0.01)
    {
      plane->normal << evecs.real()(0, evalsMin), evecs.real()(1, evalsMin),
                       evecs.real()(2, evalsMin);
      plane->min_eigen_value = evalsReal(evalsMin);
      plane->radius = sqrt(evalsReal(evalsMax));
      cout << plane->radius <<endl;
      plane->d = -(plane->normal(0) * plane->center(0) +
                  plane->normal(1) * plane->center(1) +
                  plane->normal(2) * plane->center(2));
      plane->p_center.x = plane->center(0);
      plane->p_center.y = plane->center(1);
      plane->p_center.z = plane->center(2);
      plane->p_center.normal_x = plane->normal(0);
      plane->p_center.normal_y = plane->normal(1);
      plane->p_center.normal_z = plane->normal(2);
      plane->is_plane = true;
      plane->is_update = true;
      for(auto pv: points)
        plane->plane_points.push_back(pv);
    }
    else
      plane->is_plane = false;
  }

  void init_octo_tree()
  {
    if(temp_points_.size() > points_size_threshold_)
    {
      init_plane(temp_points_, plane_ptr_);
      if(plane_ptr_->is_plane == true)
      {
        // cout<< "plane_center" << endl;
        // cout << plane_ptr_ ->center <<endl;
        // cout<< "min_eigen_value" << endl;
        // cout << plane_ptr_ ->min_eigen_value <<endl;
      }

      if(plane_ptr_->is_plane == true)
        octo_state_ = 0;
      else
      {
        octo_state_ = 1;
        cut_octo_tree();
      }
      init_octo_ = true;
      new_points_ = 0;
    }
  }

  void cut_octo_tree()
  {
    if(layer_ >= max_layer)
    {
      octo_state_ = 0;
      return;
    }
    for(size_t i = 0; i < temp_points_.size(); i++)
    {
      int xyz[3] = {0, 0, 0};
      Eigen::Vector3d pi = temp_points_[i];
      if(pi[0] > voxel_center_[0]) xyz[0] = 1;
      if(pi[1] > voxel_center_[1]) xyz[1] = 1;
      if(pi[2] > voxel_center_[2]) xyz[2] = 1;

      int leafnum = 4 * xyz[0] + 2 * xyz[1] + xyz[2];
      if(leaves_[leafnum] == nullptr)
      {
        leaves_[leafnum] = new OctoTree(layer_ + 1, 10, planer_threshold_);
        leaves_[leafnum]->layer_size_ = layer_size_;
        leaves_[leafnum]->voxel_center_[0] =
          voxel_center_[0] + (2 * xyz[0] - 1) * quater_length_;
        leaves_[leafnum]->voxel_center_[1] =
          voxel_center_[1] + (2 * xyz[1] - 1) * quater_length_;
        leaves_[leafnum]->voxel_center_[2] =
          voxel_center_[2] + (2 * xyz[2] - 1) * quater_length_;
        leaves_[leafnum]->quater_length_ = quater_length_ / 2;
      }
      leaves_[leafnum]->temp_points_.push_back(temp_points_[i]);
      leaves_[leafnum]->new_points_++;
    }
    for(uint i = 0; i < 8; i++)
      if(leaves_[i] != nullptr)
        if(leaves_[i]->temp_points_.size() > leaves_[i]->points_size_threshold_)
        {
          init_plane(leaves_[i]->temp_points_, leaves_[i]->plane_ptr_);
          if(leaves_[i]->plane_ptr_->is_plane)
            leaves_[i]->octo_state_ = 0;
          else
          {
            leaves_[i]->octo_state_ = 1;
            leaves_[i]->cut_octo_tree();
          }
          leaves_[i]->init_octo_ = true;
          leaves_[i]->new_points_ = 0;
        }
  }

  void get_plane_list(std::vector<Plane *> &plane_list)
  {
    if(plane_ptr_->is_plane)
      plane_list.push_back(plane_ptr_);
    else
      if(layer_ < max_layer)
        for(int i = 0; i < 8; i++)
          if(leaves_[i] != nullptr)
            leaves_[i]->get_plane_list(plane_list);
  }
};

class  VOXEL_LOC
{
public:


  int64_t x, y, z;

  VOXEL_LOC(int64_t vx = 0, int64_t vy = 0, int64_t vz = 0) : x(vx), y(vy), z(vz) {}

  bool operator==(const VOXEL_LOC &other) const
  {
    return (x == other.x && y == other.y && z == other.z);
  }
};
// Hash value
#define HASH_P 116101
#define MAX_N 10000000000
#define SMALL_EPS 1e-10
namespace std
{
template <> struct hash<VOXEL_LOC>
{
  size_t operator()(const VOXEL_LOC &s) const
  {
    using std::hash;
    using std::size_t;
    double cub_len = 0.125;
    long index_x, index_y, index_z;
    index_x = int(round(floor((s.x) / cub_len + SMALL_EPS)));
    index_y = int(round(floor((s.y) / cub_len + SMALL_EPS)));
    index_z = int(round(floor((s.z) / cub_len + SMALL_EPS)));
    return (((((index_z * HASH_P) % MAX_N + index_y) * HASH_P) % MAX_N) + index_x) % MAX_N;
  }
};
} // names

struct M_POINT
{
  float xyz[3];
  float intensity;
  int count = 0;
};

void down_sampling_voxel(pcl::PointCloud<pcl::PointXYZI>& pl_feat,
                         double voxel_size) 
                         {
  // int intensity = rand() % 255;
  if(voxel_size < 0.01) {
    return;
  }
  std::unordered_map<VOXEL_LOC, M_POINT> feat_map;
  uint plsize = pl_feat.size();
  cout << plsize << endl;

  for(uint i = 0; i < plsize; i++) {
    pcl::PointXYZI &p_c = pl_feat[i];
    float loc_xyz[3];
    for(int j = 0; j < 3; j++) {
      loc_xyz[j] = p_c.data[j] / voxel_size;
      if(loc_xyz[j] < 0) {
        loc_xyz[j] -= 1.0;
      }
    }

    VOXEL_LOC position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1],
                       (int64_t)loc_xyz[2]);
    auto iter = feat_map.find(position);
    if(iter != feat_map.end()) {
      iter->second.xyz[0] += p_c.x;
      iter->second.xyz[1] += p_c.y;
      iter->second.xyz[2] += p_c.z;
      iter->second.intensity += p_c.intensity;
      iter->second.count++;
    } else {
      M_POINT anp;
      anp.xyz[0] = p_c.x;
      anp.xyz[1] = p_c.y;
      anp.xyz[2] = p_c.z;
      anp.intensity = p_c.intensity;
      anp.count = 1;
      feat_map[position] = anp;
    }
  }
  plsize = feat_map.size();
  pl_feat.clear();
  pl_feat.resize(plsize);

  uint i = 0;
  for(auto iter = feat_map.begin(); iter != feat_map.end(); ++iter) {
    pl_feat[i].x = iter->second.xyz[0] / iter->second.count;
    pl_feat[i].y = iter->second.xyz[1] / iter->second.count;
    pl_feat[i].z = iter->second.xyz[2] / iter->second.count;
    pl_feat[i].intensity = iter->second.intensity / iter->second.count;
    i++;
  }
}

void down_sampling_voxel(std::vector<Eigen::Vector3d> &pl_feat,
                         double voxel_size) 
{
  std::unordered_map<VOXEL_LOC, M_POINT> feat_map;
  uint plsize = pl_feat.size();

  for(uint i = 0; i < plsize; i++) {
    Eigen::Vector3d &p_c = pl_feat[i];
    double loc_xyz[3];
    for(int j = 0; j < 3; j++) {
      loc_xyz[j] = p_c[j] / voxel_size;
      if(loc_xyz[j] < 0) {
        loc_xyz[j] -= 1.0;
      }
    }

    VOXEL_LOC position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1],
                       (int64_t)loc_xyz[2]);
    auto iter = feat_map.find(position);
    if(iter != feat_map.end()) {
      iter->second.xyz[0] += p_c[0];
      iter->second.xyz[1] += p_c[1];
      iter->second.xyz[2] += p_c[2];
      iter->second.count++;
    } else {
      M_POINT anp;
      anp.xyz[0] = p_c[0];
      anp.xyz[1] = p_c[1];
      anp.xyz[2] = p_c[2];
      anp.count = 1;
      feat_map[position] = anp;
    }
  }

  plsize = feat_map.size();
  pl_feat.resize(plsize);

  uint i = 0;
  for(auto iter = feat_map.begin(); iter != feat_map.end(); ++iter) {
    pl_feat[i][0] = iter->second.xyz[0] / iter->second.count;
    pl_feat[i][1] = iter->second.xyz[1] / iter->second.count;
    pl_feat[i][2] = iter->second.xyz[2] / iter->second.count;
    i++;
  }
}

void mergePlane(std::vector<Plane*>& origin_list, std::vector<Plane*>& merge_list)
{
    for(size_t i = 0; i < origin_list.size(); i++)
      origin_list[i]->id = 0;

    int current_id = 1;
    for(auto iter = origin_list.end() - 1; iter != origin_list.begin(); iter--)
    {
      for(auto iter2 = origin_list.begin(); iter2 != iter; iter2++)
      {
        Eigen::Vector3d normal_diff = (*iter)->normal - (*iter2)->normal;
        Eigen::Vector3d normal_add = (*iter)->normal + (*iter2)->normal;
        double dis1 = fabs((*iter)->normal(0) * (*iter2)->center(0) +
                      (*iter)->normal(1) * (*iter2)->center(1) +
                      (*iter)->normal(2) * (*iter2)->center(2) + (*iter)->d);
        double dis2 = fabs((*iter2)->normal(0) * (*iter)->center(0) +
                      (*iter2)->normal(1) * (*iter)->center(1) +
                      (*iter2)->normal(2) * (*iter)->center(2) + (*iter2)->d);
        if(normal_diff.norm() < 0.2 || normal_add.norm() < 0.2)
          if(dis1 < 0.05 && dis2 < 0.05)
          {
            if((*iter)->id == 0 && (*iter2)->id == 0)
            {
              (*iter)->id = current_id;
              (*iter2)->id = current_id;
              current_id++;
            }
            else if((*iter)->id == 0 && (*iter2)->id != 0)
              (*iter)->id = (*iter2)->id;
            else if((*iter)->id != 0 && (*iter2)->id == 0)
              (*iter2)->id = (*iter)->id;
          }
      }
    }
    std::vector<int> merge_flag;
    for(size_t i = 0; i < origin_list.size(); i++)
    {
      auto it = std::find(merge_flag.begin(), merge_flag.end(), origin_list[i]->id);
      if(it != merge_flag.end()) continue;
      
      if(origin_list[i]->id == 0)
      {
        merge_list.push_back(origin_list[i]);
        continue;
      }
      Plane* merge_plane = new Plane;
      (*merge_plane) = (*origin_list[i]);
      for(size_t j = 0; j < origin_list.size(); j++)
      {
        if(i == j) continue;
        if(origin_list[i]->id != 0)
          if(origin_list[j]->id == origin_list[i]->id)
            for(auto pv: origin_list[j]->plane_points)
              merge_plane->plane_points.push_back(pv);
      }
      merge_plane->covariance = Eigen::Matrix3d::Zero();
      merge_plane->center = Eigen::Vector3d::Zero();
      merge_plane->normal = Eigen::Vector3d::Zero();
      merge_plane->points_size = merge_plane->plane_points.size();
      merge_plane->radius = 0;
      for(auto pv : merge_plane->plane_points)
      {
        merge_plane->covariance += pv * pv.transpose();
        merge_plane->center += pv;
      }
      merge_plane->center = merge_plane->center / merge_plane->points_size;
      merge_plane->covariance = merge_plane->covariance / merge_plane->points_size -
        merge_plane->center * merge_plane->center.transpose();
      Eigen::EigenSolver<Eigen::Matrix3d> es(merge_plane->covariance);
      Eigen::Matrix3cd evecs = es.eigenvectors();
      Eigen::Vector3cd evals = es.eigenvalues();
      Eigen::Vector3d evalsReal;
      evalsReal = evals.real();
      Eigen::Matrix3f::Index evalsMin, evalsMax;
      evalsReal.rowwise().sum().minCoeff(&evalsMin);
      evalsReal.rowwise().sum().maxCoeff(&evalsMax);
      int evalsMid = 3 - evalsMin - evalsMax;
      Eigen::Vector3d evecMin = evecs.real().col(evalsMin);
      Eigen::Vector3d evecMid = evecs.real().col(evalsMid);
      Eigen::Vector3d evecMax = evecs.real().col(evalsMax);
      merge_plane->id = origin_list[i]->id;
      merge_plane->normal << evecs.real()(0, evalsMin),
        evecs.real()(1, evalsMin), evecs.real()(2, evalsMin);
      merge_plane->min_eigen_value = evalsReal(evalsMin);
      merge_plane->radius = sqrt(evalsReal(evalsMax));
      merge_plane->d = -(merge_plane->normal(0) * merge_plane->center(0) +
                        merge_plane->normal(1) * merge_plane->center(1) +
                        merge_plane->normal(2) * merge_plane->center(2));
      merge_plane->p_center.x = merge_plane->center(0);
      merge_plane->p_center.y = merge_plane->center(1);
      merge_plane->p_center.z = merge_plane->center(2);
      merge_plane->p_center.normal_x = merge_plane->normal(0);
      merge_plane->p_center.normal_y = merge_plane->normal(1);
      merge_plane->p_center.normal_z = merge_plane->normal(2);
      merge_plane->is_plane = true;
      merge_flag.push_back(merge_plane->id);
      merge_list.push_back(merge_plane);
    }
}

#endif