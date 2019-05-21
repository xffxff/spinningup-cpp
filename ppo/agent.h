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
		std::cout << &md << ", " << &model << std::endl;
	};
	auto selectAction(torch::Tensor state)
	{
		torch::NoGradGuard no_grad;
		auto out = model->forward(state);
		auto dist = std::get<0>(out);
		auto val = torch::Tensor(std::get<1>(out));
		auto act = dist.sample(1)[0];
		auto log_prob = dist.log_prob(act);
		return std::make_tuple(act, val, log_prob);
	}

	auto update(torch::Tensor obs, 
		torch::Tensor acts, 
		torch::Tensor rets, 
		torch::Tensor advs, 
		torch::Tensor old_log_pis,
		torch::Tensor old_vals)
	{
		auto out = model->forward(obs);
		auto dist = std::get<0>(out);
		auto vals = torch::Tensor(std::get<1>(out));
		auto log_pis = dist.log_prob(acts);

		auto kl = (log_pis - old_log_pis).pow(2).mean();

		auto entropy = dist.entropy().mean();
		auto ratio = torch::exp(log_pis - old_log_pis);

		auto pi_loss = -torch::min(torch::clamp(ratio, 1.0 - 0.2, 1.0 + 0.2) * advs, advs).mean();
		auto v_loss = 0.5 * (rets - vals).pow(2).mean();
		auto loss = (v_loss * 0.5 + pi_loss - entropy * 0.01);

		optimizer.zero_grad();
		loss.backward();
		optimizer.step();

		return std::make_tuple(pi_loss, v_loss, entropy, kl);
	}

};

#endif // !AGENT_H_
