import torch
from tqdm import tqdm

# TODO: Loss for maximizing sample entropy or minimizing class entropy
def train_epoch(model, optim, loader, epoch, device):
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    correct, running_loss, sample_entropy, entropy = 0, 0.0, 0.0, 0.0
    for i, (inputs, targets) in tqdm(enumerate(loader), total=len(loader)):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs, entropies, sample_entropies = model(inputs)

        loss = criterion(outputs, targets)
        optim.zero_grad()
        loss.backward()
        optim.step()

        # other stats
        running_loss += loss.item()
        _, preds = torch.max(outputs.data, 1)
        correct += (preds == targets).sum().item()
        entropy += entropies.mean().item()
        sample_entropy += sample_entropies.mean().item()

    return (running_loss/len(loader), correct/len(loader.dataset), entropy/len(loader),
            sample_entropy/len(loader))

@torch.no_grad()
def eval_model(model, loader, device, leafstats=None):
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    correct, running_loss = 0, 0.0
    leaf_samples = torch.zeros(model.n_leaves)
    leaf_correct = torch.zeros(model.n_leaves)
    leaves, all_targets, calibclasses = [], [], []
    all_leaves = torch.empty(len(loader.dataset))
    correct_indices = torch.empty(len(loader.dataset))
    for i, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs, cur_leaves = model(inputs)
        batch_size = inputs.size(0)
        all_leaves[i*128:(i*128+batch_size)] = cur_leaves

        # stats
        _, preds = torch.max(outputs.data, 1)
        running_loss += criterion(outputs, targets).item()
        correct_indices[i*128:(i*128+batch_size)] = (preds == targets)
        correct += (preds == targets).sum().item()

        # leaf stats
        if leafstats:
            leaf_samples += leafstats.sample(cur_leaves)
            leaf_correct += leafstats.correct(cur_leaves, (preds==targets))
        leaves.append(cur_leaves)
        all_targets.append(targets)

    if leafstats:
        calibclasses = leafstats.calib(torch.concat(all_targets), 
                                       torch.concat(leaves))
        leaf_correct /= (leaf_samples + 1e-10)
        leaf_samples /= len(loader.dataset)
    return (running_loss/len(loader), correct/len(loader.dataset), 
            leaf_samples, calibclasses, leaf_correct, all_leaves,
            correct_indices)

@torch.no_grad()
def eval_model_cnn(model, loader, device, leafstats=None):
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    correct, running_loss = 0, 0.0
    leaf_samples = torch.zeros(model.classifier.n_leaves)
    leaf_correct = torch.zeros(model.classifier.n_leaves)
    leaves, all_targets, calibclasses = [], [], []
    all_leaves = torch.empty(len(loader.dataset))
    correct_indices = torch.empty(len(loader.dataset))
    for i, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs, cur_leaves = model(inputs)
        batch_size = inputs.size(0)
        all_leaves[i*128:(i*128+batch_size)] = cur_leaves

        # stats
        _, preds = torch.max(outputs.data, 1)
        running_loss += criterion(outputs, targets).item()
        correct_indices[i*128:(i*128+batch_size)] = (preds == targets)
        correct += (preds == targets).sum().item()

        # leaf stats
        if leafstats:
            leaf_samples += leafstats.sample(cur_leaves)
            leaf_correct += leafstats.correct(cur_leaves, (preds==targets))
        leaves.append(cur_leaves)
        all_targets.append(targets)

    if leafstats:
        calibclasses = leafstats.calib(torch.concat(all_targets), 
                                       torch.concat(leaves))
        leaf_correct /= (leaf_samples + 1e-10)
        leaf_samples /= len(loader.dataset)
    return (running_loss/len(loader), correct/len(loader.dataset), 
            leaf_samples, calibclasses, leaf_correct, all_leaves,
            correct_indices)

# TODO: Loss for maximizing sample entropy or minimizing class entropy
# def train_epoch_warmup(model, optim, loader, epoch, device):
#     model.train()
#     n_leaves = model.n_leaves
#     criterion = torch.nn.CrossEntropyLoss()
#     correct, running_loss, sample_entropy, entropy = 0, 0.0, 0.0, 0.0
#     # correct = torch.zeros(n_leaves, dtype=torch.long)
#     for i, (inputs, targets) in tqdm(enumerate(loader), total=len(loader)):
#         inputs, targets = inputs.to(device), targets.to(device)
#         outputs = model(inputs, warmup=True)
#
#         #TODO: Check if all leaves change each pass
#         loss = criterion(outputs[:, 0], targets)
#         optim.zero_grad()
#         loss.backward()
#         optim.step()
#
#         # Stats:
#         running_loss += loss.item()
#         _, preds = torch.max(outputs[:, 0].data, 1)
#         correct += (preds == targets).sum().item()
#         # Backward for each leaf
#         # for l in range(n_leaves):
#         #     leaf_outputs = outputs[:, l]
#         #     loss = criterion(leaf_outputs, targets)
#         #     optim.zero_grad()
#         #     loss.backward(retain_graph=True)
#         #     optim.step()
#         #
#         #     # Stats:
#         #     running_loss += loss.item()
#         #     _, preds = torch.max(leaf_outputs.data, 1)
#         #     correct[l] += (preds == targets).sum().item()
#     # return (running_loss / (n_leaves*len(loader))), correct/len(loader.dataset)
#     return running_loss/len(loader), correct/len(loader.dataset)

def train_epoch_warmup(model, optim, loader, epoch, device):
    model.train()
    n_leaves = model.n_leaves
    criterion = torch.nn.CrossEntropyLoss()
    correct, running_loss, sample_entropy, entropy = 0, 0.0, 0.0, 0.0
    # correct = torch.zeros(n_leaves, dtype=torch.long)
    for i, (inputs, targets) in tqdm(enumerate(loader), total=len(loader)):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs, warmup=True)

        #TODO: Check if all leaves change each pass
        loss = criterion(outputs[:, 0], targets)
        optim.zero_grad()
        loss.backward()
        optim.step()

        # Stats:
        running_loss += loss.item()
        _, preds = torch.max(outputs[:, 0].data, 1)
        correct += (preds == targets).sum().item()
        # Backward for each leaf
        # for l in range(n_leaves):
        #     leaf_outputs = outputs[:, l]
        #     loss = criterion(leaf_outputs, targets)
        #     optim.zero_grad()
        #     loss.backward(retain_graph=True)
        #     optim.step()
        #
        #     # Stats:
        #     running_loss += loss.item()
        #     _, preds = torch.max(leaf_outputs.data, 1)
        #     correct[l] += (preds == targets).sum().item()
    # return (running_loss / (n_leaves*len(loader))), correct/len(loader.dataset)
    # return running_loss/len(loader), correct/len(loader.dataset)
    return (running_loss/len(loader), correct/len(loader.dataset), 0, 0)

# TODO: Loss for maximizing sample entropy or minimizing class entropy
def train_epoch_infav2(model, optim, loader, epoch, device):
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    correct, running_loss, sample_entropy, entropy = 0, 0.0, 0.0, 0.0
    for i, (inputs, targets) in tqdm(enumerate(loader), total=len(loader)):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)

        # Backward 
        loss = criterion(outputs, targets)
        optim.zero_grad()
        loss.backward()
        optim.step()

        # Stats:
        running_loss += loss.item()
        _, preds = torch.max(outputs.data, 1)
        correct += (preds == targets).sum().item()
    return running_loss/len(loader), correct/len(loader.dataset)

# TODO: Loss for maximizing sample entropy or minimizing class entropy
def train_epoch_infa(model, optim, loader, epoch, device, sched:bool = False):
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    correct, running_loss, sample_entropy, entropy = 0, 0.0, 0.0, 0.0
    a = (epoch+1) / 50
    s = 0
    for i, (inputs, targets) in tqdm(enumerate(loader), total=len(loader)):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs, inf_outputs, entropies, sample_entropies, mixtures = model(inputs)

        train_loss, inf_loss = criterion(outputs, targets), criterion(inf_outputs, targets)

        if sched == False:
            loss = train_loss*(1 - epoch/50) + inf_loss*(1 + epoch/50)
        else:
            if epoch > 25:
                loss = train_loss + inf_loss
            else:
                loss = train_loss

        optim.zero_grad()
        loss.backward()
        optim.step()

        # other stats
        running_loss += loss.item()
        _, preds = torch.max(outputs.data, 1)
        correct += (preds == targets).sum().item()
        entropy += entropies.mean().item()
        sample_entropy += sample_entropies.mean().item()
    # n_mean = model.node_weights.sum().item()
    # l_mean = model.w1s.sum().item()
    # import logging
    # logging.info(f"Epoch {epoch}: leaves sum {l_mean}; nodes sum {n_mean}")

    return (running_loss/len(loader), correct/len(loader.dataset), entropy/len(loader),
            sample_entropy/len(loader))

# TODO: Loss for maximizing sample entropy or minimizing class entropy
def train_epoch_afff(model, optim, loader, criterion, leaf_criterion, epoch, entropy_effect, device):
    model.train()
    correct, running_loss = 0, 0.0
    for i, (inputs, targets, leaf_targets) in tqdm(enumerate(loader), total=len(loader)):
        inputs, targets, leaf_targets = (inputs.to(device), targets.to(device), 
                                         torch.stack(leaf_targets).T.to(device))
        outputs, leaf_outputs, entropies = model(inputs)

        # back propagation
        leaf_loss = leaf_criterion(leaf_outputs, leaf_targets)*10.0
        loss = criterion(outputs, targets) + leaf_loss
        optim.zero_grad()
        loss.backward()
        optim.step()

        # other stats
        running_loss += loss.item()
        _, preds = torch.max(outputs.data, 1)
        correct += (preds == targets).sum().item()

    return running_loss/len(loader), correct/len(loader.dataset)

@torch.no_grad()
def eval_model_afff(model, loader, criterion, device, leafstats):
    model.eval()
    correct, running_loss = 0, 0.0
    leaf_samples = torch.zeros(model.n_leaves)
    leaf_correct = torch.zeros(model.n_leaves)
    leaves, all_targets = [], []
    for inputs, targets, _ in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs, cur_leaves = model(inputs)

        # stats
        _, preds = torch.max(outputs.data, 1)
        running_loss += criterion(outputs, targets).item()
        correct += (preds == targets).sum().item()

        # leaf stats
        leaf_samples += leafstats.sample(cur_leaves)
        leaf_correct += leafstats.correct(cur_leaves, (preds==targets))
        leaves.append(cur_leaves)
        all_targets.append(targets)

    return running_loss/len(loader), correct/len(loader.dataset), leaf_samples

def train_cluster(model, loader, criterion, device):
    model.train()
    correct, running_loss = 0, 0.0
    for i, (inputs, targets) in tqdm(enumerate(loader), total=len(loader)):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs_ = model(inputs)

        # back propagation
        probs = F.softmax(outputs, dim=1)  # Convert to cluster probabilities
        loss = -torch.mean(torch.sum(probs * torch.log(probs + 1e-6), dim=1))
        loss = criterion(outputs, targets) + hard_loss
        optim.zero_grad()
        loss.backward()
        optim.step()

        # other stats
        running_loss += loss.item()
        _, preds = torch.max(outputs.data, 1)
        correct += (preds == targets).sum().item()

        
        # Loss: Encourage confident assignments (maximize entropy)
        loss = -torch.mean(torch.sum(cluster_probs * torch.log(cluster_probs + 1e-6), dim=1))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')

def train_epoch_reg(model, optim, loader, epoch, sigmoid_a, sigmoid_sched, device):
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    correct, n_samples = 0, 0
    sample_entropy, entropy = 0.0, 0.0
    running_loss, running_ce, running_certainity = 0.0, 0.0, 0.0
    certainity_loss, sample_loss = torch.tensor(0.0), torch.tensor(0.0)
    sigmoid_a = 1.0
    for i, (inputs, targets) in tqdm(enumerate(loader), total=len(loader)):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs, entropies, sample_entropies = model(inputs, a=sigmoid_a)

        ce_loss = criterion(outputs, targets)
        entropy += entropies.mean().item()
        sample_entropy += sample_entropies.mean().item()
        loss = ce_loss + certainity_loss - sample_loss
        optim.zero_grad()
        loss.backward()
        optim.step()

        # other stats
        running_ce += ce_loss.item()
        running_certainity += certainity_loss.item()
        _, preds = torch.max(outputs.data, 1)
        correct += (preds == targets).sum().item()

    return (running_ce/len(loader), correct/len(loader.dataset), entropy/len(loader),
            sample_entropy/len(loader), sigmoid_a)

def train_epoch_infa_gradnorm(model, optim, optim2, weights, loader, epoch, device):
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    correct, running_loss, sample_entropy, entropy = 0, 0.0, 0.0, 0.0
    a = (epoch+1) / 50
    s = 0
    # alpha = 0.12
    alpha = 0.5 
    for i, (inputs, targets) in tqdm(enumerate(loader), total=len(loader)):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs, inf_outputs, entropies, sample_entropies = model(inputs)

        # train_loss, inf_loss = criterion(outputs, targets), criterion(inf_outputs, targets)
        loss = torch.stack((criterion(outputs, targets), criterion(inf_outputs, targets)))
        weighted_loss = weights @ loss
        optim.zero_grad()
        weighted_loss.backward(retain_graph=True)
        gw = []
        for j in range(2):
            layers = [model.w1s, model.b1s, model.w2s, model.b2s]
            # layers = [model.node_weights, model.node_biases]
            # layers += [model.node_weights, model.node_biases]
            dl = torch.autograd.grad(weights[j]*loss[j], layers, retain_graph=True, 
                                     create_graph=True)[0]
            gw.append(torch.norm(dl))
        gw = torch.stack(gw)
        loss_ratio = loss.detach() / 2
        rt = loss_ratio / loss_ratio.mean()
        gw_avg = gw.mean().detach()
        constant = (gw_avg * rt ** alpha).detach()
        gradnorm_loss = torch.abs(gw - constant).sum()
        optim2.zero_grad()
        gradnorm_loss.backward()
        optim.step()
        optim2.step()
        weights = (weights / weights.sum() * 2).detach()
        weights = torch.nn.Parameter(weights)
        # loss = (1 - a) * train_loss + a * inf_loss # EaseIn
        # loss = (1 - a**2) * train_loss + (a**2) * inf_loss # EaseIn
        # loss = (1 - (a - 1)**2) * train_loss + ((a - 1)**2) * inf_loss # EaseOut
        # loss = loss.sum()
        # optim.zero_grad()
        # loss.backward()
        # optim.step()

        # other stats
        # running_loss += loss.item()
        _, preds = torch.max(outputs.data, 1)
        correct += (preds == targets).sum().item()
        entropy += entropies.mean().item()
        sample_entropy += sample_entropies.mean().item()

    return (running_loss/len(loader), correct/len(loader.dataset), entropy/len(loader),
            sample_entropy/len(loader), weights)

# TODO: Loss for maximizing sample entropy or minimizing class entropy
def train_epoch_infa_gradnormv2(model, optim, loader, epoch, device):
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    correct, running_loss, sample_entropy, entropy = 0, 0.0, 0.0, 0.0
    a = (epoch+1) / 50
    s = 0
    model_params = [model.w1s, model.w2s, model.b1s, model.b2s, model.node_weights, 
                    model.node_biases]
    for i, (inputs, targets) in tqdm(enumerate(loader), total=len(loader)):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs, inf_outputs, entropies, sample_entropies = model(inputs)

        train_loss, inf_loss = criterion(outputs, targets), criterion(inf_outputs, targets)

        # GradNorm
        train_grad = torch.autograd.grad(train_loss, model_params, retain_graph=True)
        inf_grad = torch.autograd.grad(inf_loss, model_params, retain_graph=True)
        train_norm = sum([g.norm() for g in train_grad])
        inf_norm = sum([g.norm() for g in inf_grad])
        lambda1 = inf_norm / (train_norm + inf_norm)
        lambda2 = train_norm / (train_norm + inf_norm)

        loss = lambda1 * train_loss + lambda2 * inf_loss
        # if epoch < 10:
        #     loss = train_loss
        # else:
        #     loss = inf_loss 
        # loss = train_loss + inf_loss
        # loss = (1 - a) * train_loss + a * inf_loss # EaseIn
        # loss = (1 - a**2) * train_loss + (a**2) * inf_loss # EaseIn
        # loss = (1 - (a - 1)**2) * train_loss + ((a - 1)**2) * inf_loss # EaseOut
        optim.zero_grad()
        loss.backward()
        # optim.step()

        # other stats
        running_loss += loss.item()
        _, preds = torch.max(inf_outputs.data, 1)
        # _, preds = torch.max(outputs.data, 1)
        correct += (preds == targets).sum().item()
        entropy += entropies.mean().item()
        sample_entropy += sample_entropies.mean().item()

    return (running_loss/len(loader), correct/len(loader.dataset), entropy/len(loader),
            sample_entropy/len(loader))
