#include <vector>

#include "hdf5.h"
#include "hdf5_hl.h"

#include "caffe/layers/hdf5_output_layer.hpp"
#include "caffe/util/hdf5.hpp"
#include "caffe/util/vector_helper.hpp"

namespace caffe {

template <typename Dtype>
void HDF5OutputLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  file_name_ = this->layer_param_.hdf5_output_param().file_name();
  file_iter_ = 0;

  if( this->layer_param_.hdf5_output_param().dset_name().size() == 0) {
    // if no data set names are given, be compatible to old implementation
    dset_names_.push_back( HDF5_DATA_DATASET_NAME);
    dset_names_.push_back( HDF5_DATA_LABEL_NAME);
  }  else {
    for (int i = 0; i < this->layer_param_.hdf5_output_param().dset_name().size(); ++i) {
      dset_names_.push_back(this->layer_param_.hdf5_output_param().dset_name(i));
    }
  }
  // check if the number of dset_names matches the number of
  // bottom blobs
  CHECK_EQ( bottom.size(), dset_names_.size());
}

template <typename Dtype>
HDF5OutputLayer<Dtype>::~HDF5OutputLayer<Dtype>() {
}

template <typename Dtype>
void HDF5OutputLayer<Dtype>::SaveBlobs(const vector<Blob<Dtype>*>& bottom, bool is_GPU_data) {
  char formatted_file_name[2048];
  sprintf( formatted_file_name, file_name_.c_str(), file_iter_);
  LOG(INFO) << "Saving HDF5 file " << formatted_file_name;
  hid_t file_id = 0;
  // FileCreatPropList fplist;
  // H5Pset_obj_track_times(fplist.getId(), false);
  hid_t fcplist_id = H5Pcreate (H5P_FILE_CREATE);
  H5Pset_obj_track_times(fcplist_id, false);
  hid_t faplist_id = H5Pcreate (H5P_FILE_ACCESS);
  H5Pset_obj_track_times(faplist_id , false);
  if( file_iter_ == 0 ||
      strcmp( formatted_file_name, file_name_.c_str()) != 0) {
    // in first iteration or for differntly named files, create the files
    file_id = H5Fcreate(formatted_file_name, H5F_ACC_TRUNC, fcplist_id,
                       faplist_id);
  } else {
    // otherwise open existing file for writing
    file_id = H5Fopen(formatted_file_name, H5F_ACC_RDWR, faplist_id);
  }
  hid_t plist_id = H5Fget_create_plist(file_id);
  H5Pset_obj_track_times(plist_id, false);

  CHECK_GE(file_id, 0) << "Failed to create or reopen HDF5 file" << formatted_file_name;
  for (int i = 0; i < bottom.size(); ++i) {
    char formatted_dset_name[2048];
    sprintf( formatted_dset_name, dset_names_[i].c_str(), file_iter_);
    std::vector<int> outshape;
    for( int axis = 0; axis < bottom[i]->num_axes(); ++axis) {
      if( this->layer_param_.hdf5_output_param().squeeze() &&
          bottom[i]->shape(axis) == 1) {
        // do not append axis with length 1 to output shape
      } else {
        outshape.push_back( bottom[i]->shape(axis));
      }
    }
    LOG(INFO) << "outshape = " << toString(outshape);
    Blob<Dtype> data( outshape);
    if (is_GPU_data) {
      caffe_copy( bottom[i]->count(), bottom[i]->gpu_data(),
                  data.mutable_cpu_data());
    } else {
      caffe_copy( bottom[i]->count(), bottom[i]->cpu_data(),
                  data.mutable_cpu_data());
    }
    hdf5_save_nd_dataset(file_id, formatted_dset_name, data);
  }
  herr_t status = H5Fclose(file_id);
  CHECK_GE(status, 0) << "Failed to close HDF5 file " << file_name_;
  LOG(INFO) << "Successfully saved " << bottom.size() << " blobs";
  ++file_iter_;
}

template <typename Dtype>
void HDF5OutputLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  SaveBlobs(bottom, false);
}

template <typename Dtype>
void HDF5OutputLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  return;
}

#ifdef CPU_ONLY
STUB_GPU(HDF5OutputLayer);
#endif

INSTANTIATE_CLASS(HDF5OutputLayer);
REGISTER_LAYER_CLASS(HDF5Output);

}  // namespace caffe
