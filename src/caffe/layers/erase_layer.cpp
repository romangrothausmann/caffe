// do not rename dropout/erase with clear (because clear_param as in LayerSetUp is already used?)

#include <vector>

#include "caffe/layers/erase_layer.hpp"
#include "caffe/util/vector_helper.hpp"

namespace caffe {

template <typename Dtype>
void EraseLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NeuronLayer<Dtype>::LayerSetUp(bottom, top);
}

template <typename Dtype>
void EraseLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NeuronLayer<Dtype>::Reshape(bottom, top);
  
}

template <typename Dtype>
void EraseLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  const EraseParameter& param = this->layer_param_.erase_param();
  const Dtype* bottom_data = bottom[0]->cpu_data();

  Dtype* top_data = top[0]->mutable_cpu_data();
  caffe_copy(bottom[0]->count(), bottom_data, top_data);

  offset_ = new Dtype[3];
  size_   = new Dtype[3];
  
  caffe_rng_gaussian( 3, Dtype( top[0]->shape(1) / 2.0 ), Dtype( param.erase_random_offset_magnitude() ), offset_);
  caffe_rng_gaussian( 3, Dtype( param.erase_random_size_magnitude() ), Dtype( param.erase_random_size_magnitude() ), size_);
  
  int i= 0;
  for (int n = 0; n < top[0]->shape(0); ++n) {
    for (int c = 0; c < top[0]->shape(1); ++c) {
      for (int z = 0; z < top[0]->shape(2); ++z) {
	for (int y = 0; y < top[0]->shape(3); ++y) {
	  for (int x = 0; x < top[0]->shape(4); ++x) {
	    if( offset_[0] <= x && x < offset_[0] + size_[0] &&
		offset_[1] <= y && y < offset_[1] + size_[1] &&
		offset_[2] <= z && z < offset_[2] + size_[2] ) {
	      top_data[ i ]= 0; // erases in all n, c
	      // LOG(INFO) << "erased: " << x << "," << y << "," << z << std::endl;
	    }
	    i++;
	  }
	}
      }
    }
  }
}

template <typename Dtype>
void EraseLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
}

#ifdef CPU_ONLY
STUB_GPU(EraseLayer);
#endif

INSTANTIATE_CLASS(EraseLayer);
REGISTER_LAYER_CLASS(Erase);

}  // namespace caffe
