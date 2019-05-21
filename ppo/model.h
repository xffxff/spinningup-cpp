#ifndef MODEL_H_
#define MODEL_H_
#include <torch/torch.h>
#include <iostream>
#include "categorical.h"

using namespace torch;
struct ActorCriticImpl : nn::Module
{
	nn::Sequential actor{ nullptr }, critic{ nullptr };
	ActorCriticImpl() { std::cout << "model's default constructor is called" << std::endl; };

	ActorCriticImpl(int obs_dim, int act_dim, int hidden_size) {
		std::cout << "model's constructor is called" << std::endl;
		actor = nn::Sequential(nn::Linear(obs_dim, hidden_size),
			nn::Functional(torch::tanh),
			nn::Linear(hidden_size, act_dim));
		critic = nn::Sequential(nn::Linear(obs_dim, hidden_size),
			nn::Functional(torch::relu),
			nn::Linear(hidden_size, 1));

		register_module("actor", actor);
		register_module("critic", critic);
	}

	auto forward(torch::Tensor inp) {
		auto logits = actor->forward(inp);
		auto probs = torch::softmax(logits, -1);
		auto dist = Categorical(&probs, nullptr);
		auto vals = critic->forward(inp).squeeze();
		return std::make_tuple(dist, vals);
	}
};
TORCH_MODULE(ActorCritic);

#endif // !MODEL_H_
