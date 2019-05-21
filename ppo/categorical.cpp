#include "categorical.h"

std::vector<int64_t> Categorical::extended_shape(c10::ArrayRef<int64_t> sample_shape)
{
	std::vector<int64_t> output_shape;
	output_shape.insert(output_shape.end(),
		sample_shape.begin(),
		sample_shape.end());
	output_shape.insert(output_shape.end(),
		batch_shape.begin(),
		batch_shape.end());
	return output_shape;
}

Categorical::Categorical(const torch::Tensor * probs, const torch::Tensor * logits)
{
	if ((probs == nullptr) == (logits == nullptr))
	{
		throw std::exception();
	}

	if (probs != nullptr)
	{
		if (probs->dim() < 1)
			throw std::exception();
		this->probs = *probs / probs->sum(-1, true);
		this->probs = this->probs.clamp(1.21e-7, 1. - 1.21e-7);
		this->logits = torch::log(this->probs);
	}
	else
	{
		if (logits->dim() < 1)
			throw std::exception();
		this->logits = *logits - logits->logsumexp(-1, true);
		this->probs = torch::softmax(this->logits, -1);
	}
	param = probs != nullptr ? *probs : *logits;
	num_events = param.size(-1);
	if (param.dim() > 1)
	{
		batch_shape = param.sizes().vec();
		batch_shape.resize(batch_shape.size() - 1);
	}
}

torch::Tensor Categorical::entropy()
{
	auto p_log_p = logits * probs;
	return -p_log_p.sum(-1);
}

torch::Tensor Categorical::log_prob(torch::Tensor value)
{
	value = value.to(torch::kLong).unsqueeze(-1);
	auto broadcast_tensors = torch::broadcast_tensors({ value, logits });
	value = broadcast_tensors[0];
	value = value.narrow(-1, 0, 1);
	return broadcast_tensors[1].gather(-1, value).squeeze(-1);
}

torch::Tensor Categorical::sample(c10::ArrayRef<int64_t> sample_shape)
{
	auto ext_sample_shape = extended_shape(sample_shape);
	auto param_shape = ext_sample_shape;
	param_shape.insert(param_shape.end(), { num_events });
	auto exp_probs = probs.expand(param_shape);
	torch::Tensor probs_2d;
	if (probs.dim() == 1 || probs.size(0) == 1)
	{
		probs_2d = exp_probs.view({ -1, num_events });
	}
	else
	{
		probs_2d = exp_probs.contiguous().view({ -1, num_events });
	}
	auto sample_2d = torch::multinomial(probs_2d, 1, true);
	return sample_2d.contiguous().view(ext_sample_shape);
}
