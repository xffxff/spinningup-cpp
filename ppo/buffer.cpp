#include "buffer.h"

PPOBuffer::PPOBuffer(int od, int ad, int s, double g, double l) : obs_dim(od), act_dim(ad), size(s), gamma(g), lam(l)
{
}

void PPOBuffer::store(torch::Tensor obs, int act, float rew, float done, float val, float log_pi)
{
	obs_buf.push_back(obs);
	act_buf.push_back(act);
	rew_buf.push_back(rew);
	done_buf.push_back(done);
	val_buf.push_back(val);
	log_pi_buf.push_back(log_pi);
}

auto PPOBuffer::get(float last_val) -> std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
{
	assert(rew_buf.size() == size);

	val_buf.push_back(last_val);

	std::vector<float> adv_buf(rew_buf.size());
	float last_gae_lam = 0;
	float last_ret = last_val;
	for (int i = size - 1; i >= 0; --i)
	{
		auto next_not_done = 1.0 - done_buf[i];
		auto delta = rew_buf[i] + gamma * val_buf[i + 1] * next_not_done - val_buf[i];
		last_gae_lam = delta + gamma * lam * next_not_done * last_gae_lam;
		adv_buf[i] = last_gae_lam;
	}

	val_buf.pop_back();
	auto obs_tch = torch::stack(obs_buf);
	auto act_tch = torch::from_blob(act_buf.data(), { size }, torch::dtype(torch::kInt32));
	auto val_tch = torch::from_blob(val_buf.data(), { size }, torch::dtype(torch::kFloat32));
	auto log_pi_tch = torch::from_blob(log_pi_buf.data(), { size }, torch::dtype(torch::kFloat32));
	auto adv_tch = torch::from_blob(adv_buf.data(), { size }, torch::dtype(torch::kFloat32));

	auto ret_tch = adv_tch + val_tch;
	adv_tch = (adv_tch - adv_tch.mean()) / (adv_tch.std() + 1e-8);
	obs_buf.clear();
	act_buf.clear();
	rew_buf.clear();
	done_buf.clear();
	val_buf.clear();
	log_pi_buf.clear();
	return std::make_tuple(obs_tch, act_tch, ret_tch, adv_tch, log_pi_tch, val_tch);
}