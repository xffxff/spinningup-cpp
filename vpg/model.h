#ifndef MODEL_H_
#define MODEL_H_
#include <torch/torch.h>
#include <iostream>

using namespace torch;
struct ActorCriticImpl : nn::Module
{
	nn::Sequential actor{ nullptr }, critic{ nullptr };
	ActorCriticImpl() { std::cout << "model's default constructor is called" << std::endl; };

	ActorCriticImpl(int obs_dim, int act_dim, int hidden_size) {
		std::cout << "model's constructor is called" << std::endl;
		actor = nn::Sequential(nn::Linear(obs_dim, hidden_size),
			nn::Functional(torch::relu),
			nn::Linear(hidden_size, act_dim));
		critic = nn::Sequential(nn::Linear(obs_dim, hidden_size),
			nn::Functional(torch::relu),
			nn::Linear(hidden_size, 1));

		register_module("actor", actor);
		register_module("critic", critic);
	}

	auto forward(torch::Tensor inp) {
		auto acts = actor->forward(inp);
		auto vals = critic->forward(inp);
		return std::make_tuple(torch::softmax(acts, -1), vals);
	}
};
TORCH_MODULE(ActorCritic);

#endif // !MODEL_H_
