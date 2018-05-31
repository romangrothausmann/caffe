#ifndef CAFFE_ERASE_LAYER_HPP_
#define CAFFE_ERASE_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/neuron_layer.hpp"

namespace caffe {

/**
 */
template <typename Dtype>
class EraseLayer : public NeuronLayer<Dtype> {
 public:
  /**
   * @param param provides EraseParameter erase_param,
   *     with EraseLayer options:
   *   - erase_ratio (\b optional, default 0.5).
   */
  explicit EraseLayer(const LayerParameter& param)
      : NeuronLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Erase"; }

 protected:
  /**
   */
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  Dtype* offset_;
  Dtype* size_;    
};

}  // namespace caffe

#endif  // CAFFE_ERASE_LAYER_HPP_
