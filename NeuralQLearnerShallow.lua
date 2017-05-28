--[[
Copyright (c) 2014 Google Inc.
See LICENSE file for full terms of limited license.
]]

if not dqn then
    require 'initenv'
end

local nql = torch.class('dqn.NeuralQLearner')


function nql:__init(args)
    self.state_dim  = args.state_dim -- State dimensionality.
    self.actions    = args.actions
    self.n_actions  = #self.actions
    self.verbose    = args.verbose
    self.best       = args.best

    --- epsilon annealing
    self.ep_start   = args.ep or 1
    self.ep         = self.ep_start -- Exploration probability.
    self.ep_end     = args.ep_end or self.ep
    self.ep_endt    = args.ep_endt or 1000000

    ---- learning rate annealing
    self.lr_start       = args.lr or 0.01 --Learning rate.
    self.lr             = self.lr_start
    self.lr_end         = args.lr_end or self.lr
    self.lr_endt        = args.lr_endt or 1000000
    self.wc             = args.wc or 0  -- L2 weight cost.
    self.minibatch_size = args.minibatch_size or 1
    self.valid_size     = args.valid_size or 500

    --- Q-learning parameters
    self.discount       = args.discount or 0.99 --Discount factor.
    self.update_freq    = args.update_freq or 1
    -- Number of points to replay per learning step.
    self.n_replay       = args.n_replay or 1
    -- Number of steps after which learning starts.
    self.learn_start    = args.learn_start or 0
     -- Size of the transition table.
    self.replay_memory  = args.replay_memory or 1000000
    self.hist_len       = args.hist_len or 1
    self.rescale_r      = args.rescale_r
    self.max_reward     = args.max_reward
    self.min_reward     = args.min_reward
    self.clip_delta     = args.clip_delta
    self.target_q       = args.target_q
    self.bestq          = 0

    self.gpu            = args.gpu

    self.ncols          = args.ncols or 1  -- number of color channels in input
    self.input_dims     = args.input_dims or {self.hist_len*self.ncols, 84, 84}
    self.preproc        = args.preproc  -- name of preprocessing network
    self.histType       = args.histType or "linear"  -- history type to use
    self.histSpacing    = args.histSpacing or 1
    self.nonTermProb    = args.nonTermProb or 1
    self.bufferSize     = args.bufferSize or 512

    self.transition_params = args.transition_params or {}

    self.network    = args.network or self:createNetwork()

    -- check whether there is a network file
    local network_function
    if not (type(self.network) == 'string') then
        error("The type of the network provided in NeuralQLearner" ..
              " is not a string!")
    end

    local msg, err = pcall(require, self.network)
    if not msg then
        -- try to load saved agent
        local err_msg, exp = pcall(torch.load, self.network)
        if not err_msg then
            error("Could not find network file ")
        end
        if self.best and exp.best_model then
            self.network = exp.best_model
        else
            self.network = exp.model
        end
    else
        print('Creating Agent Network from ' .. self.network)
        self.network = err
        self.network = self:network()
    end

    if self.gpu and self.gpu >= 0 then
        self.network:cuda()
    else
        self.network:float()
    end

    -- Load preprocessing network.
    if not (type(self.preproc == 'string')) then
        error('The preprocessing is not a string')
    end
    msg, err = pcall(require, self.preproc)
    if not msg then
        error("Error loading preprocessing net")
    end
    self.preproc = err
    self.preproc = self:preproc()
    self.preproc:float()

    if self.gpu and self.gpu >= 0 then
        self.network:cuda()
        self.tensor_type = torch.CudaTensor
    else
        self.network:float()
        self.tensor_type = torch.FloatTensor
    end

    -- Create transition table.
    ---- assuming the transition table always gets floating point input
    ---- (Foat or Cuda tensors) and always returns one of the two, as required
    ---- internally it always uses ByteTensors for states, scaling and
    ---- converting accordingly
    local transition_args = {
        stateDim = self.state_dim, numActions = self.n_actions,
        histLen = self.hist_len, gpu = self.gpu,
        maxSize = self.replay_memory, histType = self.histType,
        histSpacing = self.histSpacing, nonTermProb = self.nonTermProb,
        bufferSize = self.bufferSize
    }

    self.transitions = dqn.TransitionTable(transition_args)

    self.numSteps = 0 -- Number of perceived states.
    self.lastState = nil
    self.lastAction = nil
    self.v_avg = 0 -- V running average.
    self.tderr_avg = 0 -- TD error running average.

    self.q_max = 1
    self.r_max = 1

    self.w, self.dw = self.network:getParameters()
    self.dw:zero()

    self.deltas = self.dw:clone():fill(0)

    self.tmp= self.dw:clone():fill(0)
    self.g  = self.dw:clone():fill(0)
    self.g2 = self.dw:clone():fill(0)

    if self.target_q then
        self.target_network = self.network:clone()
    end

    ---------------------- Shallow_Updates_for_Deep_RL Addition ------------------------
	self.net_type    = 'dqn' -- 'dqn' or 'ddqn'
	self.SRL_method  = 'fqi' -- SRL method to replace the last layer: lstdq, fqi, or none
	self.lambda      = 1     -- Regularizer value for the SRL method
	self.between_SRL = 50    -- number of steps between SRL updates = (self.between_SRL * self.target_q)
	------------------------------------------------------------------------------------
end


function nql:reset(state)
    if not state then
        return
    end
    self.best_network = state.best_network
    self.network = state.model
    self.w, self.dw = self.network:getParameters()
    self.dw:zero()
    self.numSteps = 0
    print("RESET STATE SUCCESFULLY")
end


function nql:preprocess(rawstate)
    if self.preproc then
        return self.preproc:forward(rawstate:float())
                    :clone():reshape(self.state_dim)
    end

    return rawstate
end


function nql:getQUpdate(args)
    local s, a, r, s2, term, delta
    local q, q2, q2_max

    s = args.s
    a = args.a
    r = args.r
    s2 = args.s2
    term = args.term

    -- The order of calls to forward is a bit odd in order
    -- to avoid unnecessary calls (we only need 2).

    -- delta = r + (1-terminal) * gamma * max_a Q(s2, a) - Q(s, a)
    term = term:clone():float():mul(-1):add(1)

    local target_q_net
    if self.target_q then
        target_q_net = self.target_network
    else
        target_q_net = self.network
    end

    -- Compute max_a Q(s_2, a).
	---------------------- Shallow_Updates_for_Deep_RL Addition ------------------------
    if self.net_type == 'ddqn' then
      q2_tmp, q2_idxs = self.network:forward(s2):float():max(2)
      q2_vals = target_q_net:forward(s2):float()
      q2_max = q2_vals:gather(2, q2_idxs) -- keeping as q2_max for consistency
    elseif self.net_type == 'dqn' then
      q2_max = target_q_net:forward(s2):float():max(2)
    else
	  print('Got illegal argument for net_type. Only dqn and ddqn are supported!')
	  collectgarbage()
	  os.exit()
	end
	------------------------------------------------------------------------------------

    -- Compute q2 = (1-terminal) * gamma * max_a Q(s2, a)
    q2 = q2_max:clone():mul(self.discount):cmul(term)

    delta = r:clone():float()

    if self.rescale_r then
        delta:div(self.r_max)
    end
    delta:add(q2)

    -- q = Q(s,a)
    local q_all = self.network:forward(s):float()
    q = torch.FloatTensor(q_all:size(1))
    for i=1,q_all:size(1) do
        q[i] = q_all[i][a[i]]
    end
    delta:add(-1, q)

    if self.clip_delta then
        delta[delta:ge(self.clip_delta)] = self.clip_delta
        delta[delta:le(-self.clip_delta)] = -self.clip_delta
    end

    local targets = torch.zeros(self.minibatch_size, self.n_actions):float()
    for i=1,math.min(self.minibatch_size,a:size(1)) do
        targets[i][a[i]] = delta[i]
    end

    if self.gpu >= 0 then targets = targets:cuda() end

    return targets, delta, q2_max
end


function nql:qLearnMinibatch()
    -- Perform a minibatch Q-learning update:
    -- w += alpha * (r + gamma max Q(s2,a2) - Q(s,a)) * dQ(s,a)/dw
    assert(self.transitions:size() > self.minibatch_size)

    local s, a, r, s2, term = self.transitions:sample(self.minibatch_size)

    local targets, delta, q2_max = self:getQUpdate{s=s, a=a, r=r, s2=s2,
        term=term, update_qmax=true}

    -- zero gradients of parameters
    self.dw:zero()

    -- get new gradient
    self.network:backward(s, targets)

    -- add weight cost to gradient
    self.dw:add(-self.wc, self.w)

    -- compute linearly annealed learning rate
    local t = math.max(0, self.numSteps - self.learn_start)
    self.lr = (self.lr_start - self.lr_end) * (self.lr_endt - t)/self.lr_endt +
                self.lr_end
    self.lr = math.max(self.lr, self.lr_end)

    -- use gradients
    self.g:mul(0.95):add(0.05, self.dw)
    self.tmp:cmul(self.dw, self.dw)
    self.g2:mul(0.95):add(0.05, self.tmp)
    self.tmp:cmul(self.g, self.g)
    self.tmp:mul(-1)
    self.tmp:add(self.g2)
    self.tmp:add(0.01)
    self.tmp:sqrt()

    -- accumulate update
    self.deltas:mul(0):addcdiv(self.lr, self.dw, self.tmp)
    self.w:add(self.deltas)
end


function nql:sample_validation_data()
    local s, a, r, s2, term = self.transitions:sample(self.valid_size)
    self.valid_s    = s:clone()
    self.valid_a    = a:clone()
    self.valid_r    = r:clone()
    self.valid_s2   = s2:clone()
    self.valid_term = term:clone()
end


function nql:compute_validation_statistics()
    local targets, delta, q2_max = self:getQUpdate{s=self.valid_s,
        a=self.valid_a, r=self.valid_r, s2=self.valid_s2, term=self.valid_term}

    self.v_avg = self.q_max * q2_max:mean()
    self.tderr_avg = delta:clone():abs():mean()
end


function nql:perceive(reward, rawstate, terminal, testing, testing_ep)
    -- Preprocess state (will be set to nil if terminal)
    local state = self:preprocess(rawstate):float()
    local curState

    if self.max_reward then
        reward = math.min(reward, self.max_reward)
    end
    if self.min_reward then
        reward = math.max(reward, self.min_reward)
    end
    if self.rescale_r then
        self.r_max = math.max(self.r_max, reward)
    end

    self.transitions:add_recent_state(state, terminal)

    local currentFullState = self.transitions:get_recent()

    --Store transition s, a, r, s'
    if self.lastState and not testing then
        self.transitions:add(self.lastState, self.lastAction, reward,
                             self.lastTerminal, priority)
    end

    if self.numSteps == self.learn_start+1 and not testing then
        self:sample_validation_data()
    end

    curState= self.transitions:get_recent()
    curState = curState:resize(1, unpack(self.input_dims))

    -- Select action
    local actionIndex = 1
    if not terminal then
        actionIndex = self:eGreedy(curState, testing_ep)
    end

    self.transitions:add_recent_action(actionIndex)

    --Do some Q-learning updates
    if self.numSteps > self.learn_start and not testing and
        self.numSteps % self.update_freq == 0 then
        for i = 1, self.n_replay do
            self:qLearnMinibatch()
        end
    end

    if not testing then
        self.numSteps = self.numSteps + 1
    end

    self.lastState = state:clone()
    self.lastAction = actionIndex
    self.lastTerminal = terminal

    ---------------------- Shallow_Updates_for_Deep_RL Addition ------------------------
	if self.target_q and self.numSteps % self.target_q == 1 then
      if self.numSteps > self.replay_memory and self.numSteps % (self.between_SRL*self.target_q) == 1 then
          if self.SRL_method == 'lstdq' then
              self:lstdq_update()
          elseif self.SRL_method == 'fqi' then
              self:fitted_q_update()
          elseif self.SRL_method ~= 'none' then
		          print('Got illegal argument for SRL_method. Only lstdq, fqi, or none are supported!')
	            collectgarbage()
	            os.exit()
		      end
      end
      self.target_network = self.network:clone()
  end
	------------------------------------------------------------------------------------

    if not terminal then
        return actionIndex
    else
        return 0
    end
end


function nql:eGreedy(state, testing_ep)
    self.ep = testing_ep or (self.ep_end +
                math.max(0, (self.ep_start - self.ep_end) * (self.ep_endt -
                math.max(0, self.numSteps - self.learn_start))/self.ep_endt))
    -- Epsilon greedy
    if torch.uniform() < self.ep then
        return torch.random(1, self.n_actions)
    else
        return self:greedy(state)
    end
end


function nql:greedy(state)
    -- Turn single state into minibatch.  Needed for convolutional nets.
    if state:dim() == 2 then
        assert(false, 'Input must be at least 3D')
        state = state:resize(1, state:size(1), state:size(2))
    end

    if self.gpu >= 0 then
        state = state:cuda()
    end

    local q = self.network:forward(state):float():squeeze()
    local maxq = q[1]
    local besta = {1}

    -- Evaluate all other actions (with random tie-breaking)
    for a = 2, self.n_actions do
        if q[a] > maxq then
            besta = { a }
            maxq = q[a]
        elseif q[a] == maxq then
            besta[#besta+1] = a
        end
    end
    self.bestq = maxq

    local r = torch.random(1, #besta)

    self.lastAction = besta[r]

    return besta[r]
end


function nql:createNetwork()
    local n_hid = 128
    local mlp = nn.Sequential()
    mlp:add(nn.Reshape(self.hist_len*self.ncols*self.state_dim))
    mlp:add(nn.Linear(self.hist_len*self.ncols*self.state_dim, n_hid))
    mlp:add(nn.Rectifier())
    mlp:add(nn.Linear(n_hid, n_hid))
    mlp:add(nn.Rectifier())
    mlp:add(nn.Linear(n_hid, self.n_actions))

    return mlp
end


function nql:_loadNet()
    local net = self.network
    if self.gpu then
        net:cuda()
    else
        net:float()
    end
    return net
end


function nql:init(arg)
    self.actions = arg.actions
    self.n_actions = #self.actions
    self.network = self:_loadNet()
    -- Generate targets.
    self.transitions:empty()
end


function nql:report()
    print(get_weight_norms(self.network))
    print(get_grad_norms(self.network))
end


---------------------- Shallow_Updates_for_Deep_RL Addition ------------------------
function nql:lstdq_update()
  print('performing lstdq_update')

  -- initialization
  local N          = self.replay_memory-10
  local n_features = 512
  local n_actions  = self.n_actions
  local gamma      = self.discount

  local f1     = self.network:clone()
  local orig_w = f1.modules[11].weight:float()
  local orig_b = f1.modules[11].bias:float()

  local f2    = self.network:clone()
  local f_w   = f2.modules[11].weight:double()
  local f_b   = f2.modules[11].bias:double()
  local prior = torch.Tensor((n_features+1)*n_actions,1):zero():double()
  prior[{{1, n_features*n_actions}, 1}] = f_w:reshape(n_features*n_actions,1)
  prior[{{n_features*n_actions+1, (n_features+1)*n_actions}, 1}] = f_b:reshape(n_actions,1)

  local A = torch.CudaTensor((n_features+1)*n_actions,(n_features+1)*n_actions):zero()
  local b = torch.CudaTensor((n_features+1)*n_actions,1):zero()

  local i = 0
  local phi_
  local Nphi_
  local tmp
  local last_term = 0

  while i < (N-1) do

    i=i+1

	-- generate sample
    local s, action, reward, s2, terminal_t1 = self.transitions:get(i)
    s = s:float():div(255):resize(1, unpack(self.input_dims)):cuda()
    s2 = s2:float():div(255):resize(1, unpack(self.input_dims)):cuda()

    if i == 1 or last_term == 1 then
      self.network:forward(s)
      tmp = self.network.modules[10].output:float()
      phi_ = tmp:clone()
    else
      phi_= Nphi_:clone()
    end
    self.network:forward(s2)
    tmp = self.network.modules[10].output:float()
    Nphi_ = tmp:clone()
    last_term = terminal_t1

    -- generate next action according to network's policy
    local res = torch.mm(orig_w, Nphi_:reshape(n_features,1))
    local res2 = res:add(orig_b):reshape(n_actions)
    local Nmax = res2[1]
    local Naction = 1
    for a = 2, n_actions do
      if res2[a] > Nmax then
        Nmax = res2[a]
        Naction = a
      end
    end

	-- build feature vectors
    phi     = torch.CudaTensor((n_features+1)*n_actions,1):zero()
    Nphi    = torch.CudaTensor((n_features+1)*n_actions,1):zero()
    phi[{{1+(action-1)*n_features,n_features+(action-1)*n_features},1}] = phi_
    phi[{{n_features*n_actions+action},1}] = 1
    Nphi[{{1+(Naction-1)*n_features,n_features+(Naction-1)*n_features},1}] = Nphi_
    Nphi[{{n_features*n_actions+Naction},1}] = 1

    -- structures maintenance
    b_ = (reward*phi):div(N)
    b:add(b_)
    if (terminal_t1==0) then -- next state not terminal
      A_ = (torch.mm(phi, ((phi-gamma*Nphi):transpose(1,2)))):div(N)
      A:add(A_)
    else -- next state is terminal
      A_ = (torch.mm(phi, ((phi):transpose(1,2)))):div(N)
      A:add(A_)
      i = i+1
    end
  end

  -- generate solution
  A:add(self.lambda*(torch.eye((1+n_features)*n_actions):cuda()))
  local w = torch.gels(b:double() + prior:double()*self.lambda, A:double()):cuda()
  local w_copy  = w:clone()
  local weights = ((w_copy:narrow(1,1,n_features*n_actions)):reshape(n_actions,n_features))
  local biases  = ((w_copy:narrow(1,n_features*n_actions+1,n_actions)):reshape(n_actions))

  -- replace last layer
  self.network.modules[11].weight = weights:clone()
  self.network.modules[11].bias   = biases:clone()

end


function nql:fitted_q_update()
  print('performing fitted_q_update')

  -- initialization
  local N          = self.replay_memory-10
  local n_features = 512
  local n_actions  = self.n_actions
  local gamma      = self.discount

  local f1     = self.network:clone()
  local orig_w = f1.modules[11].weight:float()
  local orig_b = f1.modules[11].bias:float()

  local f2    = self.network:clone()
  local f_w   = f2.modules[11].weight:double()
  local f_b   = f2.modules[11].bias:double()
  local prior = torch.Tensor((n_features+1)*n_actions,1):zero():double()
  prior[{{1, n_features*n_actions}, 1}] = f_w:reshape(n_features*n_actions,1)
  prior[{{n_features*n_actions+1, (n_features+1)*n_actions}, 1}] = f_b:reshape(n_actions,1)

  local A = torch.CudaTensor((n_features+1)*n_actions,(n_features+1)*n_actions):zero()
  local b = torch.CudaTensor((n_features+1)*n_actions,1):zero()

  local i = 0
  local phi_
  local Nphi_
  local tmp
  local last_term = 0


  while i < (N-1) do

    i=i+1

	-- generate sample
    local s, action, reward, s2, terminal_t1 = self.transitions:get(i)
    s = s:float():div(255):resize(1, unpack(self.input_dims)):cuda()
    s2 = s2:float():div(255):resize(1, unpack(self.input_dims)):cuda()

    if i == 1 or last_term == 1 then
      self.network:forward(s)
      tmp = self.network.modules[10].output:float()
      phi_ = tmp:clone()
    else
      phi_= Nphi_:clone()
    end
    self.network:forward(s2)
    tmp = self.network.modules[10].output:float()
    Nphi_ = tmp:clone()
    last_term = terminal_t1

    -- generate next action according to network's policy
    local res = torch.mm(orig_w, Nphi_:reshape(n_features,1))
    local res2 = res:add(orig_b):reshape(n_actions)
    local Nmax = res2[1]
    local Naction = 1
    for a = 2, n_actions do
      if res2[a] > Nmax then
        Nmax = res2[a]
        Naction = a
      end
    end

	-- build feature vectors
    phi     = torch.CudaTensor((n_features+1)*n_actions,1):zero()
    phi[{{1+(action-1)*n_features,n_features+(action-1)*n_features},1}] = phi_
    phi[{{n_features*n_actions+action},1}] = 1

    -- structures maintenance
    A_ = (torch.mm(phi, ((phi):transpose(1,2)))):div(N)
    A:add(A_)
    if (terminal_t1==0) then -- next state not terminal
      b_ = ((reward + gamma*Nmax)*phi):div(N)
      b:add(b_)
    else -- next state terminal
      b_ = (reward*phi):div(N)
      b:add(b_)
      i = i+1
    end
  end

  -- generate solution
  A:add(self.lambda*(torch.eye((1+n_features)*n_actions):cuda()))
  local w = torch.gels(b:double() + prior:double()*self.lambda, A:double()):cuda()
  local w_copy  = w:clone()
  local weights = ((w_copy:narrow(1,1,n_features*n_actions)):reshape(n_actions,n_features))
  local biases  = ((w_copy:narrow(1,n_features*n_actions+1,n_actions)):reshape(n_actions))

  -- replace last layer
  self.network.modules[11].weight = weights:clone()
  self.network.modules[11].bias   = biases:clone()

end
------------------------------------------------------------------------------------
