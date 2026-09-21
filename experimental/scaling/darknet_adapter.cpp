#include <initializer_list>
#include <cmath>
#include <cstddef>
#include <cuda_runtime.h>
#include "darknet.h"
extern "C" {
network parse_network_cfg_custom(char *, int, int);
void push_convolutional_layer(layer);
void cuda_set_device(int);
void load_weights(network *, char *);
void *yolo_open(const char *cfg, const char *weights) {
 cuda_set_device(0);
 network *net=new network(parse_network_cfg_custom(const_cast<char*>(cfg),1,1));
 load_weights(net,const_cast<char*>(weights));
 fuse_conv_batchnorm(*net);
 for(int i=0;i<net->n;i++)if(net->layers[i].type==CONVOLUTIONAL)push_convolutional_layer(net->layers[i]);
 return net;
}
int yolo_infer(void *handle,float *image,float *output,size_t capacity) {
 network net=*static_cast<network*>(handle);
 network_predict(net,image);
 if(cudaDeviceSynchronize()!=cudaSuccess)return 1;
 size_t offset=0;
 for(int divisor: {8,16,32})for(int i=0;i<net.n;i++) {
  layer l=net.layers[i]; if(l.type!=YOLO||l.w!=net.w/divisor)continue;
  if(l.classes!=80)return 2;
  for(int a=0;a<l.n;a++)for(int y=0;y<l.h;y++)for(int x=0;x<l.w;x++) {
   if(offset+85>capacity)return 3;
   int cell=y*l.w+x,area=l.w*l.h; size_t base=a*85*area;
   output[offset++]=(x+l.output[base+cell])/l.w;
   output[offset++]=(y+l.output[base+area+cell])/l.h;
   output[offset++]=expf(l.output[base+2*area+cell])*l.biases[2*l.mask[a]]/net.w;
   output[offset++]=expf(l.output[base+3*area+cell])*l.biases[2*l.mask[a]+1]/net.h;
   for(int attr=4;attr<85;attr++)output[offset++]=l.output[base+attr*area+cell];
  }
 }
 return offset==capacity?0:4;
}
}
