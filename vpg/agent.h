#ifndef AGENT_H_
#define AGENT_H_
#include <torch/torch.h>
#include "model.h"
#include <iostream>
#include <vector>

class Agent
{
private:
	ActorCritic & model;
	torch::optim::Optimizer & optimizer;
public:
	Agent(ActorCritic & md, torch::optim::Optimizer & opt) : model(md), optimizer(opt) { 
		std::cout << &md << ", " << &model << std::endl; };
	auto selectAction(torch::Tensor state)
	{
		auto out = model->forward(state);
		auto probs = torch::Tensor(std::get<0>(out));
		auto val = torch::Tensor(std::get<1>(out));
		auto act = probs.multinomial(1)[0].item<int32_t>();
		auto p = probs / probs.sum(-1, true);
		auto log_prob = p[act].log();
		return std::make_tuple(act, val, log_prob);
	}

	auto update(std::vector<torch::Tensor> log_probs_vec,
		std::vector<torch::Tensor> vals_vec,
		std::vector<float> rews_vec,
		std::vector<float> dones_vec)
	{
		auto rets_vec = get_discout_rets(rews_vec, dones_vec);
		auto rets_tch = torch::from_blob(
			rets_vec.data(), { static_cast<int64_t>(rets_vec.size()) });
		rets_tch = (rets_tch - rets_tch.mean()) / (rets_tch.std() + 1e-8);
		/*std::vector<torch::Tensor> policy_loss, value_loss;
		for (auto i = 0U; i < log_probs_vec.size(); ++i)
		{
			auto adv = rets_tch[i] - vals_vec[i].item<float>();
			policy_loss.push_back(-adv * log_probs_vec[i]);
			value_loss.push_back(
				torch::smooth_l1_loss(vals_vec[i], torch::ones(1) * rets_tch[i]));
		}
		auto p_loss = torch::stack(policy_loss).sum();
		auto v_loss = torch::stack(value_loss).sum();
		auto loss = p_loss + v_loss;*/

		auto log_probs_tch = torch::stack(log_probs_vec);
		auto vals_tch = torch::cat(vals_vec);
		auto adv = (rets_tch - vals_tch).detach();
		auto p_loss = (-adv * log_probs_tch).sum();
		auto v_loss = torch::smooth_l1_loss(vals_tch, rets_tch).sum();
		auto loss = p_loss + v_loss;

		optimizer.zero_grad();
		loss.backward();
		optimizer.step();

		std::cout << ", " << "policy_loss: " << p_loss.item<float>()
			<< ", " << "value_loss: " << v_loss.item<float>() << std::endl;
	}

	auto get_discout_rets(std::vector<float> rews_vec, std::vector<float> dones_vec)->std::vector<float>
	{
		double ret = 0.;
		for (int i = rews_vec.size() - 1; i >= 0; --i)
		{
			ret = rews_vec[i] + 0.99 * ret * (1 - dones_vec[i]);
			rews_vec[i] = ret;
		}
		return rews_vec;
	}
};

#endif // !AGENT_H_
