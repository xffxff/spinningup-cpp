#include "cartpole.h"
#include <random>
#include <iostream>
#include <vector>
#include "model.h"
#include "agent.h"
#include "buffer.h"


const int epochs = 500;
const double gamma = 0.99;
const double lam = 0.95;
const int train_epoch_len = 1000;
const int obs_dim = 4;
const int act_dim = 2;
const int hidden_size = 128;
const int batch_size = 128;

using namespace torch;
int main()
{

	CartPole env;
	auto model = ActorCritic(obs_dim, act_dim, hidden_size);
	//auto model = std::make_shared<ActorCritic>(obs_dim, act_dim, hidden_size);
	auto optimizer = torch::optim::Adam(model->parameters(), 1e-3);
	auto agent = Agent(model, optimizer);
	auto buffer = PPOBuffer(obs_dim, act_dim, train_epoch_len, gamma, lam);

	std::vector<double> rets, pi_loss, v_loss, entropy, kl;

	auto collect_rollouts = [&] {
		double ep_rew = 0.;
		env.reset();
		auto state = env.getState();

		for (int t = 0; t < train_epoch_len; ++t)
		{
			auto out = agent.selectAction(state);
			auto act = std::get<0>(out).item<int32_t>();
			auto val = std::get<1>(out).item<float>();
			auto log_prob = std::get<2>(out).item<float>();

			env.step(act);
			auto next_state = env.getState();
			auto rew = env.getReward();
			auto done = env.isDone();

			ep_rew += rew;

			buffer.store(state, act, rew, done, val, log_prob);
			state = next_state;
			if (done)
			{
				rets.push_back(ep_rew);
				ep_rew = 0.;
				env.reset();
				state = env.getState();
			}
		}
		auto out = agent.selectAction(state);
		auto last_val = std::get<1>(out).item<float>();
		return last_val;
	};

	auto run_train_phase = [&] {
		auto last_val = collect_rollouts();
		auto data = buffer.get(last_val);

		auto obs_tch = std::get<0>(data);
		auto act_tch = std::get<1>(data);
		auto ret_tch = std::get<2>(data);
		auto adv_tch = std::get<3>(data);
		auto log_pi_tch = std::get<4>(data);
		auto val_tch = std::get<5>(data);

		for (int i = 0; i < 3; ++i) {
			auto random_idx = torch::randperm(train_epoch_len, torch::dtype(torch::kLong));
			int batch_num = train_epoch_len / batch_size;
			for (int j = 0; j < batch_num; ++j) {
				auto sample_idx = random_idx.narrow(0, j * batch_size, batch_size);
				auto obs_batch = obs_tch.index(sample_idx);
				auto act_batch = act_tch.index(sample_idx);
				auto ret_batch = ret_tch.index(sample_idx);
				auto adv_batch = adv_tch.index(sample_idx);
				auto log_pi_batch = log_pi_tch.index(sample_idx);
				auto val_batch = val_tch.index(sample_idx);

				auto out = agent.update(obs_batch, act_batch, ret_batch, adv_batch, log_pi_batch, val_batch);
				pi_loss.push_back(std::get<0>(out).item<float>());
				v_loss.push_back(std::get<1>(out).item<float>());
				entropy.push_back(std::get<2>(out).item<float>());
				kl.push_back(std::get<3>(out).item<float>());
			}
		}

	};
	
	for (int i = 0; i < epochs; ++i)
	{
		run_train_phase();
		std::cout << "ep_ret_mean: " << std::accumulate(rets.begin(), rets.end(), 0.) / rets.size() << "   " 
			<< "pi_loss: " << std::accumulate(pi_loss.begin(), pi_loss.end(), 0.) / pi_loss.size() << "   "
			<< "v_loss: "<< std::accumulate(v_loss.begin(), v_loss.end(), 0.) / v_loss.size() << "   "
			<< "entropy: " << std::accumulate(entropy.begin(), entropy.end(), 0.) / entropy.size() << "   "
			<< "kl: " << std::accumulate(kl.begin(), kl.end(), 0.) / kl.size() << std::endl;
		rets.clear();
	}

}