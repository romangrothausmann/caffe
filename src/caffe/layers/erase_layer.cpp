// do not rename dropout/erase with clear (because clear_param as in LayerSetUp is already used?)

#include <vector>

#include "caffe/layers/erase_layer.hpp"

namespace caffe {

template <typename Dtype>
void EraseLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NeuronLayer<Dtype>::LayerSetUp(bottom, top);
  threshold_ = this->layer_param_.erase_param().erase_ratio();
  DCHECK(threshold_ > 0.);
  DCHECK(threshold_ < 1.);
  uint_thres_ = static_cast<unsigned int>(UINT_MAX * threshold_);
}

template <typename Dtype>
void EraseLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NeuronLayer<Dtype>::Reshape(bottom, top);
}

template <typename Dtype>
void EraseLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();

  Dtype* top_data = top[0]->mutable_cpu_data();
  caffe_copy(bottom[0]->count(), bottom_data, top_data);
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
