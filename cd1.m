function ret = cd1(rbm_w, visible_data)
  visible_data = sample_bernoulli(visible_data);

% <rbm_w> is a matrix of size <number of hidden units> by <number of visible units>
% <visible_data> is a (possibly but not necessarily binary) matrix of size <number of visible units> by <number of data cases>
% The returned value is the gradient approximation produced by CD-1. It's of the same shape as <rbm_w>.

% sample_bernoulli() was called with a matrix of size 100 by 1. 
% sample_bernoulli() was called with a matrix of size 256 by 1. 
% sample_bernoulli() was called with a matrix of size 100 by 1. 
% Describing a matrix of size 100 by 256. 
% The mean of the elements is -0.160742. The sum of the elements is -4115.000000


  h0 = visible_state_to_hidden_probabilities(rbm_w, visible_data);
  h0_s = sample_bernoulli(h0);
  v1 = hidden_state_to_visible_probabilities(rbm_w,h0_s);
  v1_s = sample_bernoulli(v1);
  h1 = visible_state_to_hidden_probabilities(rbm_w, v1_s);
%  h1_s = sample_bernoulli(h1);
  ret = configuration_goodness_gradient(visible_data,h0_s) .- configuration_goodness_gradient(v1_s,h1);
  ret = ret';
  % fprintf('h0 is a matrix of size %d by %d. \n', size(h0, 1), size(h0, 2));
  % fprintf('v1 is a matrix of size %d by %d. \n', size(v1, 1), size(v1, 2));
  % fprintf('h1 is a matrix of size %d by %d. \n', size(h1, 1), size(h1, 2));
  % fprintf('ret is a matrix of size %d by %d. \n', size(ret, 1), size(ret, 2));
end
