#include "cartpole.h"
#include <random>
#include <iostream>
#include <vector>
#include "model.h"
#include "agent.h"

using namespace torch;
int main()
{
	std::vector<torch::Tensor> log_probs_buf, vals_buf;
	std::vector<float> rews_buf, dones_buf;

	CartPole env;
	auto model = ActorCritic(4, 2, 128);
	auto optimizer = torch::optim::Adam(model->parameters(), 1e-3);
	auto agent = Agent(model, optimizer);

	for (int epoch = 0; ; ++epoch)
	{
		env.reset();
		auto state = env.getState();
		int episodes = 0;
		for (int t = 0; t < 1000; ++t)
		{
			auto out = agent.selectAction(state);
			auto act = std::get<0>(out);
			auto val = std::get<1>(out);
			auto log_prob = std::get<2>(out);

			log_probs_buf.push_back(log_prob);
			vals_buf.push_back(val);

			env.step(act);
			state = env.getState();
			auto rew = env.getReward();
			auto done = env.isDone();
			
			rews_buf.push_back(rew);
			dones_buf.push_back(done);
			if (done)
			{
				episodes++;
				env.reset();
				state = env.getState();
			}
		}

		std::cout << "ep_rew_mean: " << std::accumulate(rews_buf.begin(), rews_buf.end(), 0.0) / episodes;
		agent.update(log_probs_buf, vals_buf, rews_buf, dones_buf);
		log_probs_buf.clear();
		vals_buf.clear();
		rews_buf.clear();
		dones_buf.clear();

	}
}