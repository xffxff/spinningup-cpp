#include <torch/torch.h>
#include <vector>


class PPOBuffer
{
private:
	int obs_dim, act_dim, size;
	double gamma, lam;
	std::vector<torch::Tensor> obs_buf;
	std::vector<int> act_buf;
	std::vector<float> val_buf, rew_buf, log_pi_buf, done_buf;
public:
	PPOBuffer(int od, int ad, int s, double g, double l);
	void store(torch::Tensor obs, int act, float rew, float done, float val, float log_pi);
	auto get(float last_val)->std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>;
};