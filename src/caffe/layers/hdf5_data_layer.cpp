/*
TODO:
- load file in a separate thread ("prefetch")
- can be smarter about the memcpy call instead of doing it row-by-row
  :: use util functions caffe_copy, and Blob->offset()
  :: don't forget to update hdf5_daa_layer.cu accordingly
- add ability to shuffle filenames if flag is set
*/
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <vector>

#include "hdf5.h"
#include "hdf5_hl.h"
#include "stdint.h"

#include "caffe/layers/hdf5_data_layer.hpp"
#include "caffe/util/hdf5.hpp"
#include "caffe/util/vector_helper.hpp"

namespace caffe {

template <typename Dtype>
HDF5DataLayer<Dtype>::~HDF5DataLayer<Dtype>() { }

// Load data and label from HDF5 filename into the class property blobs.
template <typename Dtype>
void HDF5DataLayer<Dtype>::LoadHDF5FileData(const char* filename) {
  DLOG(INFO) << "Loading HDF5 file: " << filename;
  hid_t file_id = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
  if (file_id < 0) {
    LOG(FATAL) << "Failed opening HDF5 file: " << filename;
  }

  int top_size = this->layer_param_.top_size();
  hdf_blobs_.resize(top_size);

  const int MIN_DATA_DIM = 1;
  const int MAX_DATA_DIM = INT_MAX;

  for (int i = 0; i < top_size; ++i) {
    hdf_blobs_[i] = shared_ptr<Blob<Dtype> >(new Blob<Dtype>());
    hdf5_load_nd_dataset(file_id, this->layer_param_.top(i).c_str(),
        MIN_DATA_DIM, MAX_DATA_DIM, hdf_blobs_[i].get());
  }

  herr_t status = H5Fclose(file_id);
  CHECK_GE(status, 0) << "Failed to close HDF5 file: " << filename;

  // MinTopBlobs==1 guarantees at least one top blob
  CHECK_GE(hdf_blobs_[0]->num_axes(), 1) << "Input must have at least 1 axis.";
  const int num = hdf_blobs_[0]->shape(0);
  for (int i = 1; i < top_size; ++i) {
    CHECK_EQ(hdf_blobs_[i]->shape(0), num);
  }
  // Default to identity permutation.
  data_permutation_.clear();
  data_permutation_.resize(hdf_blobs_[0]->shape(0));
  for (int i = 0; i < hdf_blobs_[0]->shape(0); i++)
    data_permutation_[i] = i;

  // Shuffle if needed.
  if (this->layer_param_.hdf5_data_param().shuffle()) {
    std::random_shuffle(data_permutation_.begin(), data_permutation_.end());
    DLOG(INFO) << "Successully loaded " << hdf_blobs_[0]->shape(0)
               << " rows (shuffled)";
  } else {
    DLOG(INFO) << "Successully loaded " << hdf_blobs_[0]->shape(0) << " rows";
  }
}

template <typename Dtype>
void HDF5DataLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Refuse transformation parameters since HDF5 is totally generic.
  CHECK(!this->layer_param_.has_transform_param()) <<
      this->type() << " does not transform data.";
  // Read the source to parse the filenames.
  const string& source = this->layer_param_.hdf5_data_param().source();
  LOG(INFO) << "Loading list of HDF5 filenames from: " << source;
  hdf_filenames_.clear();
  std::ifstream source_file(source.c_str());
  if (source_file.is_open()) {
    std::string line;
    while (source_file >> line) {
      hdf_filenames_.push_back(line);
    }
  } else {
    LOG(FATAL) << "Failed to open source file: " << source;
  }
  source_file.close();
  num_files_ = hdf_filenames_.size();
  current_file_ = 0;
  last_file_ = -1;
  last_row_ = -1;
  LOG(INFO) << "Number of HDF5 files: " << num_files_;
  CHECK_GE(num_files_, 1) << "Must have at least 1 HDF5 filename listed in "
    << source;

  // check the shapes of all datasets, if they are equal, and if the
  // number of images per blob is divisible by the batch size
  const int batch_size = this->layer_param_.hdf5_data_param().batch_size();
  int num_datasets = top.size();
  dataset_shapes_.resize( num_files_ * num_datasets);
  files_have_consistent_shapes_ = true;
  hdf_blobs_divisible_by_batch_size_ = true;
  for (int fi = 0; fi < num_files_; ++fi) {
    std::string msg;
    std::string filename = hdf_filenames_[fi];
    hid_t file_id = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    msg += filename + ":";
    for (int di = 0; di < num_datasets; ++di) {
      std::vector<hsize_t> shape =
          hdf5_get_dataset_shape( file_id, this->layer_param_.top(di).c_str());
      dataset_shapes_[fi * num_datasets + di] = shape;
      if( di > 0) {
        if( shape.size() != dataset_shapes_[di].size()) {
          LOG(FATAL) << filename << " dataset " << this->layer_param_.top(di).c_str() << toString( shape) << " has different number of axes.";
        }
        if( shape[0] % batch_size != 0) {
          hdf_blobs_divisible_by_batch_size_ = false;
        }
        for (int j = 1; j < shape.size(); ++j) {
          if( shape[j] != dataset_shapes_[di][j]) {
            files_have_consistent_shapes_ = false;
          }
        }
      }
      msg += std::string("  ") + this->layer_param_.top(di).c_str() + " " + toString( shape);
    }
    LOG(INFO) << msg;
    herr_t status = H5Fclose(file_id);
    CHECK_GE(status, 0) << "Failed to close HDF5 file: " << filename;
  }
  LOG(INFO) << "files_have_consistent_shapes: "
            << files_have_consistent_shapes_;
  LOG(INFO) << "hdf_blobs_divisible_by_batch_size: "
            << hdf_blobs_divisible_by_batch_size_;

  if( files_have_consistent_shapes_ == false
      && hdf_blobs_divisible_by_batch_size_ == false) {
    LOG(FATAL) <<
        "Cannot work with these files! The dataset must have either\n"
        "the same spatial shapes, or the batch sizes in the HDF5 data\n"
        "sets must be divisible by the requested training batch size!";
  }

  file_permutation_.clear();
  file_permutation_.resize(num_files_);
  // Default to identity permutation.
  for (int i = 0; i < num_files_; i++) {
    file_permutation_[i] = i;
  }

  // Shuffle if needed.
  if (this->layer_param_.hdf5_data_param().shuffle()) {
    std::random_shuffle(file_permutation_.begin(), file_permutation_.end());
  }

  // initialize the line counter.
  current_row_ = 0;
}

template <typename Dtype>
void HDF5DataLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                   const vector<Blob<Dtype>*>& top) {
  const int batch_size = this->layer_param_.hdf5_data_param().batch_size();

  // index of current file
  int fi = file_permutation_[current_file_];

  // Reshape blobs.
  const int top_size = this->layer_param_.top_size();
  vector<int> top_shape;
  for (int i = 0; i < top_size; ++i) {
    top_shape.resize( dataset_shapes_[fi * top_size + i].size());
    top_shape[0] = batch_size;
    for (int j = 1; j < top_shape.size(); ++j) {
      top_shape[j] = dataset_shapes_[fi * top_size + i][j];
    }
    top[i]->Reshape(top_shape);
  }
}

template <typename Dtype>
void HDF5DataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int batch_size = this->layer_param_.hdf5_data_param().batch_size();
  for (int i = 0; i < batch_size; ++i) {
    // if at the begin of a new file, load the data
    if(file_permutation_[current_file_] != last_file_){ // only load H5 if file-index (file name) changed)
      if( current_row_ == 0) {
        LoadHDF5FileData(
            hdf_filenames_[file_permutation_[current_file_]].c_str());
      }
      DLOG(INFO) << "last_file_: " << last_file_;
    }

    // copy data to top blobs
    if(data_permutation_[current_row_] != last_row_ || // only copy H5 data if row-index (sub-dataset) changed
       file_permutation_[current_file_] != last_file_){ // OR if file-index (file name) changed)
      for (int j = 0; j < this->layer_param_.top_size(); ++j) {
        int data_dim = top[j]->count() / top[j]->shape(0);
        caffe_copy(data_dim,
            &hdf_blobs_[j]->cpu_data()[data_permutation_[current_row_]
              * data_dim], &top[j]->mutable_cpu_data()[i * data_dim]);
      }
      DLOG(INFO) << "last_row_: " << last_row_;
    }

    last_row_ = data_permutation_[current_row_]; // rember last row
    last_file_ = file_permutation_[current_file_]; // rember last file

    // advance index to next "row", possibly go to next file
    ++current_row_;
    if (current_row_ == hdf_blobs_[0]->shape(0)) {
      if (num_files_ > 1) {
        ++current_file_;
        if (current_file_ == num_files_) {
          current_file_ = 0;
          if (this->layer_param_.hdf5_data_param().shuffle()) {
            std::random_shuffle(file_permutation_.begin(),
                                file_permutation_.end());
          DLOG(INFO) << "Looping around to first file.";
          }
        }
      }
      current_row_ = 0;
      if (this->layer_param_.hdf5_data_param().shuffle())
        std::random_shuffle(data_permutation_.begin(), data_permutation_.end());
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU_FORWARD(HDF5DataLayer, Forward);
#endif

INSTANTIATE_CLASS(HDF5DataLayer);
REGISTER_LAYER_CLASS(HDF5Data);

}  // namespace caffe
