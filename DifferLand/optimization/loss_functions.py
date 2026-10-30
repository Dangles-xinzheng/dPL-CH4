import jax
import jax.numpy as jnp
import jax.nn as jnn

def sum_of_error(predicted, true, mask, weight=1):
    return jnp.sum((predicted-true)**2 * mask) * weight

def patch_of_error(predicted, true, mask, weight=1):
    return jnp.sum(((jnp.mean(predicted * mask, axis=0) - jnp.mean(true * mask, axis=0))**2) * jnp.mean(mask, axis=0)) * weight

    
def loss_996_mse_with_grace(params, initial_pools, predictors, met, labels, batch_forward, pfn, warm_up=12, batch_size=320):
    result = batch_forward(params, initial_pools, predictors, met).squeeze()
    sif_loss = sum_of_error(result[:, warm_up:, pfn.SIF], labels[:, warm_up:, 0], labels[:, warm_up:, 1], weight=500)
    lai_loss = sum_of_error(result[:, warm_up:, pfn.lai], labels[:, warm_up:, 4], labels[:, warm_up:, 5], weight=50)
    modeled_agb = result[:, warm_up:, pfn.next_labile_pool] + result[:, warm_up:, pfn.next_foliar_pool] + result[:, warm_up:, pfn.next_wood_pool]
    observed_agb = labels[:, warm_up:, 8]
    agb_mask = labels[:, warm_up:, 9]
    
    modeled_agb_monthly = jnp.mean(modeled_agb.reshape(batch_size, -1, 12), axis=2)
    observed_agb_monthly=jnp.mean(observed_agb.reshape(batch_size, -1, 12), axis=2)
    agb_mask_monthly = jnp.prod(agb_mask.reshape(agb_mask.shape[0], -1, 12), axis=2)
    agb_loss = sum_of_error(modeled_agb_monthly, observed_agb_monthly, agb_mask_monthly, weight=5e-6)
    nbe_loss = patch_of_error(result[:, warm_up:, pfn.nee], labels[:, warm_up:, 2], labels[:, warm_up:, 3], weight=3000)
    predicted_water_pool_mean = jnp.sum(result[:, warm_up:, pfn.next_water_pool] * labels[:, warm_up:, 7], axis=1) / (jnp.sum(labels[:, warm_up:, 7], axis=1) + 0.01)
    water_loss = patch_of_error((result[:, warm_up:, pfn.next_water_pool].T-predicted_water_pool_mean).T, labels[:, warm_up:, 6], labels[:, warm_up:, 7], weight=0.2)

    return sif_loss+lai_loss+agb_loss+nbe_loss+water_loss

def loss_996_mse_without_grace(params, initial_pools, predictors, met, labels, batch_forward, pfn, warm_up=12, batch_size=320):
    result = batch_forward(params, initial_pools, predictors, met).squeeze()
    sif_loss = sum_of_error(result[:, warm_up:, pfn.SIF], labels[:, warm_up:, 0], labels[:, warm_up:, 1], weight=400)
    lai_loss = sum_of_error(result[:, warm_up:, pfn.lai], labels[:, warm_up:, 4], labels[:, warm_up:, 5], weight=40)
    modeled_agb = result[:, warm_up:, pfn.next_labile_pool] + result[:, warm_up:, pfn.next_foliar_pool] + result[:, warm_up:, pfn.next_wood_pool]
    observed_agb = labels[:, warm_up:, 8]
    agb_mask = labels[:, warm_up:, 9]
    
    modeled_agb_monthly = jnp.mean(modeled_agb.reshape(batch_size, -1, 12), axis=2)
    observed_agb_monthly=jnp.mean(observed_agb.reshape(batch_size, -1, 12), axis=2)
    agb_mask_monthly = jnp.prod(agb_mask.reshape(agb_mask.shape[0], -1, 12), axis=2)
    agb_loss = sum_of_error(modeled_agb_monthly, observed_agb_monthly, agb_mask_monthly, weight=5e-6)
    nbe_loss = patch_of_error(result[:, warm_up:, pfn.nee], labels[:, warm_up:, 2], labels[:, warm_up:, 3], weight=3000)
    return sif_loss+lai_loss+agb_loss+nbe_loss

def compute_nse_vec(observed, modeled, mask, epsilon=0.001):
    numerator = jnp.sum((observed - modeled) ** 2 * mask, axis=1)
    observed_mean = jnp.sum(observed * mask, axis=1) / (jnp.sum(mask, axis=1) + epsilon)
    denominator = jnp.sum(((observed.T - observed_mean)**2).T * mask, axis=1) + epsilon
    nse_vec = 1- numerator/denominator
    weighted_nse = jnp.sum(nse_vec * jnp.sum(mask, axis=1)) / (jnp.sum(mask) + epsilon)
    return weighted_nse

def compute_nnse_vec(observed, modeled, mask):
    """
    Compute normalized Nash-Sutcliffe Efficiency (NNSE) for vectorized inputs.
    
    NNSE = 1 / (2 - NSE)
    
    To prevent division by zero or negative values, we clip the denominator
    to a small positive value (epsilon) when it's too close to zero.
    """
    nse = compute_nse_vec(observed, modeled, mask)
    denominator = 2 - nse
    
    # Prevent division by zero or negative values
    # Clip denominator to [epsilon, inf] to ensure numerical stability
    epsilon = 1e-6
    denominator = jnp.maximum(denominator, epsilon)
    
    return 1 / denominator

def loss_991_mse_with_grace(params, initial_pools, predictors, met, labels, pfn, batch_forward, warm_up=12, batch_size=320):
    result = batch_forward(params, initial_pools, predictors, met).squeeze()
    sif_loss = sum_of_error(result[:, warm_up:, pfn.SIF], labels[:, warm_up:, 0], labels[:, warm_up:, 1], weight=400)
    lai_loss = sum_of_error(result[:, warm_up:, pfn.lai], labels[:, warm_up:, 4], labels[:, warm_up:, 5], weight=40)
    modeled_agb = result[:, warm_up:, pfn.next_labile_pool] + result[:, warm_up:, pfn.next_foliar_pool] + result[:, warm_up:, pfn.next_wood_pool]
    observed_agb = labels[:, warm_up:, 8]
    agb_mask = labels[:, warm_up:, 9]
    
    modeled_agb_monthly = jnp.mean(modeled_agb.reshape(batch_size, -1, 12), axis=2)
    observed_agb_monthly=jnp.mean(observed_agb.reshape(batch_size, -1, 12), axis=2)
    agb_mask_monthly = jnp.prod(agb_mask.reshape(agb_mask.shape[0], -1, 12), axis=2)
    agb_loss = sum_of_error(modeled_agb_monthly, observed_agb_monthly, agb_mask_monthly, weight=5e-6)
    nbe_loss = patch_of_error(result[:, warm_up:, pfn.nee], labels[:, warm_up:, 2], labels[:, warm_up:, 3], weight=3000)

    predicted_water_pool_mean = jnp.sum((result[:, warm_up:, pfn.next_paw_pool] + result[:, warm_up:, pfn.next_puw_pool]) * labels[:, warm_up:, 7], axis=1) / (jnp.sum(labels[:, warm_up:, 7], axis=1) + 0.01)
    water_loss = patch_of_error(((result[:, warm_up:, pfn.next_paw_pool] + result[:, warm_up:, pfn.next_puw_pool]).T-predicted_water_pool_mean).T, labels[:, warm_up:, 6], labels[:, warm_up:, 7], weight=0.2)

    return sif_loss+lai_loss+agb_loss+nbe_loss+water_loss

def loss_991_mse_without_grace(params, initial_pools, predictors, met, labels, batch_forward, pfn, warm_up=12, batch_size=320):
    result = batch_forward(params, initial_pools, predictors, met).squeeze()
    sif_loss = sum_of_error(result[:, warm_up:, pfn.SIF], labels[:, warm_up:, 0], labels[:, warm_up:, 1], weight=400)
    lai_loss = sum_of_error(result[:, warm_up:, pfn.lai], labels[:, warm_up:, 4], labels[:, warm_up:, 5], weight=40)
    modeled_agb = result[:, warm_up:, pfn.next_labile_pool] + result[:, warm_up:, pfn.next_foliar_pool] + result[:, warm_up:, pfn.next_wood_pool]
    observed_agb = labels[:, warm_up:, 8]
    agb_mask = labels[:, warm_up:, 9]
    
    modeled_agb_monthly = jnp.mean(modeled_agb.reshape(batch_size, -1, 12), axis=2)
    observed_agb_monthly=jnp.mean(observed_agb.reshape(batch_size, -1, 12), axis=2)
    agb_mask_monthly = jnp.prod(agb_mask.reshape(agb_mask.shape[0], -1, 12), axis=2)
    agb_loss = sum_of_error(modeled_agb_monthly, observed_agb_monthly, agb_mask_monthly, weight=5e-6)
    nbe_loss = patch_of_error(result[:, warm_up:, pfn.nee], labels[:, warm_up:, 2], labels[:, warm_up:, 3], weight=3000)
    return sif_loss+lai_loss+agb_loss+nbe_loss

def loss_991_nnse_with_grace(params, initial_pools, predictors, met, labels, batch_forward, pfn, warm_up=12, batch_size=320, epsilon=0.001):
    result = batch_forward(params, initial_pools, predictors, met).squeeze()
    sif_loss = compute_nnse_vec(result[:, warm_up:, pfn.SIF], labels[:, warm_up:, 0], labels[:, warm_up:, 1])
    lai_loss = compute_nnse_vec(result[:, warm_up:, pfn.lai], labels[:, warm_up:, 4], labels[:, warm_up:, 5])
    modeled_agb = result[:, warm_up:, pfn.next_labile_pool] + result[:, warm_up:, pfn.next_foliar_pool] + result[:, warm_up:, pfn.next_wood_pool]
    observed_agb = labels[:, warm_up:, 8]
    agb_mask = labels[:, warm_up:, 9]
    modeled_agb_monthly = jnp.mean(modeled_agb.reshape(batch_size, -1, 12), axis=2)
    observed_agb_monthly=jnp.mean(observed_agb.reshape(batch_size, -1, 12), axis=2)
    agb_mask_monthly = jnp.prod(agb_mask.reshape(agb_mask.shape[0], -1, 12), axis=2)
    agb_loss = compute_nnse_vec(modeled_agb_monthly, observed_agb_monthly, agb_mask_monthly)
    nbe_loss = compute_nnse_vec(result[:, warm_up:, pfn.nee], labels[:, warm_up:, 2], labels[:, warm_up:, 3])
    predicted_water_pool_mean = jnp.sum((result[:, warm_up:, pfn.next_paw_pool] + result[:, warm_up:, pfn.new_puw_pool]) * labels[:, warm_up:, 7], axis=1) / (jnp.sum(labels[:, warm_up:, 7], axis=1) + epsilon)
    water_loss = compute_nnse_vec(((result[:, warm_up:, pfn.next_paw_pool] + result[:, warm_up:, pfn.next_puw_pool]).T-predicted_water_pool_mean).T, labels[:, warm_up:, 6], labels[:, warm_up:, 7])

    return -(sif_loss + lai_loss + agb_loss + nbe_loss + water_loss)

def loss_991_nnse_without_grace(params, initial_pools, predictors, met, labels, batch_forward, pfn, warm_up=12, batch_size=320):
    result = batch_forward(params, initial_pools, predictors, met).squeeze()
    sif_loss = compute_nnse_vec(result[:, warm_up:, pfn.SIF], labels[:, warm_up:, 0], labels[:, warm_up:, 1])
    lai_loss = compute_nnse_vec(result[:, warm_up:, pfn.lai], labels[:, warm_up:, 4], labels[:, warm_up:, 5])
    modeled_agb = result[:, warm_up:, pfn.next_labile_pool] + result[:, warm_up:, pfn.next_foliar_pool] + result[:, warm_up:, pfn.next_wood_pool]
    observed_agb = labels[:, warm_up:, 8]
    agb_mask = labels[:, warm_up:, 9]
    modeled_agb_monthly = jnp.mean(modeled_agb.reshape(batch_size, -1, 12), axis=2)
    observed_agb_monthly=jnp.mean(observed_agb.reshape(batch_size, -1, 12), axis=2)
    agb_mask_monthly = jnp.prod(agb_mask.reshape(agb_mask.shape[0], -1, 12), axis=2)
    agb_loss = compute_nnse_vec(modeled_agb_monthly, observed_agb_monthly, agb_mask_monthly)
    nbe_loss = compute_nnse_vec(result[:, warm_up:, pfn.nee], labels[:, warm_up:, 2], labels[:, warm_up:, 3])

    return -(sif_loss + lai_loss + agb_loss + nbe_loss)


def loss_996_nnse_with_grace(params, initial_pools, predictors, met, labels, batch_forward, warm_up=12, batch_size=320, epsilon=0.001):
    result = batch_forward(params, initial_pools, predictors, met).squeeze()
    sif_loss = compute_nnse_vec(result[:, warm_up:, 39], labels[:, warm_up:, 0], labels[:, warm_up:, 1])
    lai_loss = compute_nnse_vec(result[:, warm_up:, 0], labels[:, warm_up:, 4], labels[:, warm_up:, 5])
    modeled_agb = result[:, warm_up:, 32] + result[:, warm_up:, 33] + result[:, warm_up:, 35]
    observed_agb = labels[:, warm_up:, 8]
    agb_mask = labels[:, warm_up:, 9]
    modeled_agb_monthly = jnp.mean(modeled_agb.reshape(batch_size, -1, 12), axis=2)
    observed_agb_monthly=jnp.mean(observed_agb.reshape(batch_size, -1, 12), axis=2)
    agb_mask_monthly = jnp.prod(agb_mask.reshape(agb_mask.shape[0], -1, 12), axis=2)
    agb_loss = compute_nnse_vec(modeled_agb_monthly, observed_agb_monthly, agb_mask_monthly)
    nbe_loss = compute_nnse_vec(result[:, warm_up:, 31], labels[:, warm_up:, 2], labels[:, warm_up:, 3])
    predicted_water_pool_mean = jnp.sum(result[:, warm_up:, 38]  * labels[:, warm_up:, 7], axis=1) / (jnp.sum(labels[:, warm_up:, 7], axis=1) + epsilon)
    water_loss = compute_nnse_vec((result[:, warm_up:, 38] .T-predicted_water_pool_mean).T, labels[:, warm_up:, 6], labels[:, warm_up:, 7])

    return -(sif_loss + lai_loss + agb_loss + nbe_loss + water_loss)


def loss_996_nnse_without_grace(params, initial_pools, predictors, met, labels, batch_forward, warm_up=12, batch_size=320, epsilon=0.001):
    result = batch_forward(params, initial_pools, predictors, met).squeeze()
    sif_loss = compute_nnse_vec(result[:, warm_up:, 39], labels[:, warm_up:, 0], labels[:, warm_up:, 1])
    lai_loss = compute_nnse_vec(result[:, warm_up:, 0], labels[:, warm_up:, 4], labels[:, warm_up:, 5])
    modeled_agb = result[:, warm_up:, 32] + result[:, warm_up:, 33] + result[:, warm_up:, 35]
    observed_agb = labels[:, warm_up:, 8]
    agb_mask = labels[:, warm_up:, 9]
    modeled_agb_monthly = jnp.mean(modeled_agb.reshape(batch_size, -1, 12), axis=2)
    observed_agb_monthly=jnp.mean(observed_agb.reshape(batch_size, -1, 12), axis=2)
    agb_mask_monthly = jnp.prod(agb_mask.reshape(agb_mask.shape[0], -1, 12), axis=2)
    agb_loss = compute_nnse_vec(modeled_agb_monthly, observed_agb_monthly, agb_mask_monthly)
    nbe_loss = compute_nnse_vec(result[:, warm_up:, 31], labels[:, warm_up:, 2], labels[:, warm_up:, 3])

    return -(sif_loss + lai_loss + agb_loss + nbe_loss)

"""
def nse_score(targets, predictions, mask):
    return 1-(np.sum((targets-predictions)**2 * mask)/np.sum((targets-np.sum(targets)/np.sum())**2))

def loss_996_nnse_no_grace(params, initial_pools, predictors, met, labels, batch_forward, warm_up=12, batch_size=320):
    result = batch_forward(params, initial_pools, predictors, met).squeeze()
    sif_loss = sum_of_error(result[:, warm_up:, 39], labels[:, warm_up:, 0], labels[:, warm_up:, 1], weight=100)
    lai_loss = sum_of_error(result[:, warm_up:, 0], labels[:, warm_up:, 4], labels[:, warm_up:, 5], weight=5)
    modeled_agb = result[:, warm_up:, 32] + result[:, warm_up:, 33] + result[:, warm_up:, 35]
    observed_agb = labels[:, warm_up:, 8]
    agb_mask = labels[:, warm_up:, 9]
    
    modeled_agb_monthly = jnp.mean(modeled_agb.reshape(batch_size, -1, 12), axis=2)
    observed_agb_monthly=jnp.mean(observed_agb.reshape(batch_size, -1, 12), axis=2)
    agb_mask_monthly = jnp.prod(agb_mask.reshape(agb_mask.shape[0], -1, 12), axis=2)
    agb_loss = sum_of_error(modeled_agb_monthly, observed_agb_monthly, agb_mask_monthly, weight=5e-6)
    nbe_loss = patch_of_error(result[:, warm_up:, 31], labels[:, warm_up:, 2], labels[:, warm_up:, 3], weight=4000)
    return sif_loss+lai_loss+agb_loss+nbe_loss
"""

# methods for local models
def compute_nse(observed, modeled, mask):
    """
    Compute Nash-Sutcliffe Efficiency (NSE).
    
    NSE = 1 - sum((observed - modeled)^2 * mask) / sum((observed - mean(observed))^2 * mask)
    
    When mask_sum = 0 (no valid data), returns 0.0 (neutral value).
    To ensure gradient stability when mask_sum = 0, we use jax.lax.cond to conditionally
    compute NSE only when there's data, preventing NaN gradients.
    
    Uses jax.lax.cond for numerical stability when the mask is empty.
    """
    mask_sum = jnp.sum(mask)
    
    def compute_nse_with_data():
        """Compute NSE when mask_sum > 0"""
        numerator = jnp.sum((observed - modeled) ** 2 * mask)
        observed_mean = jnp.sum(observed * mask) / mask_sum
        denominator = jnp.sum((observed - observed_mean)**2 * mask)
        # Handle case where denominator is 0 (perfect fit)
        return jnp.where(denominator > 0, 1 - numerator/denominator, 1.0)
    
    def return_zero():
        """Return 0.0 when mask_sum = 0"""
        return jnp.array(0.0)
    
    # Use jax.lax.cond so a zero mask does not evaluate NaN-producing branches.
    return jax.lax.cond(mask_sum > 0, compute_nse_with_data, return_zero)


def compute_nnse(observed, modeled, mask):
    """
    Compute normalized Nash-Sutcliffe Efficiency (NNSE).
    
    NNSE = 1 / (2 - NSE)
    
    To prevent division by zero or negative values in gradient computation,
    we clip the denominator to a small positive value (epsilon) when it's too close to zero.
    """
    nse = compute_nse(observed, modeled, mask)
    denominator = 2 - nse
    
    # Prevent division by zero or negative values
    # Clip denominator to [epsilon, inf] to ensure numerical stability
    epsilon = 1e-6
    denominator = jnp.maximum(denominator, epsilon)
    
    return 1 / denominator

def compute_nnse_eval(output_matrix, target_matrix, train_sel, test_sel, reco=False):
        
    gpp_train_nnse = compute_nnse(target_matrix[train_sel, 0], output_matrix[train_sel, 1], target_matrix[train_sel, 1])
    if reco:
        nee_reco_train_nnse = compute_nnse(target_matrix[train_sel, 8], output_matrix[train_sel, 1] + output_matrix[train_sel, 21], target_matrix[train_sel, 9])     
    else:
        nee_reco_train_nnse = compute_nnse(target_matrix[train_sel, 2], output_matrix[train_sel, 21], target_matrix[train_sel, 3])
    et_train_nnse = compute_nnse(target_matrix[train_sel, 4], output_matrix[train_sel, 2], target_matrix[train_sel, 5])
    lai_train_nnse = compute_nnse(target_matrix[train_sel, 6], output_matrix[train_sel, 0], target_matrix[train_sel, 7])
    
    gpp_test_nnse = compute_nnse(target_matrix[test_sel, 0], output_matrix[test_sel, 1], target_matrix[test_sel, 1])
    if reco:
        nee_reco_test_nnse = compute_nnse(target_matrix[test_sel, 8], output_matrix[test_sel, 1] + output_matrix[test_sel, 21], target_matrix[test_sel, 9])     
    else:
        nee_reco_test_nnse = compute_nnse(target_matrix[test_sel, 2], output_matrix[test_sel, 21], target_matrix[test_sel, 3])
    et_test_nnse = compute_nnse(target_matrix[test_sel, 4], output_matrix[test_sel, 2], target_matrix[test_sel, 5])
    lai_test_nnse = compute_nnse(target_matrix[test_sel, 6], output_matrix[test_sel, 0], target_matrix[test_sel, 7])
    
    return gpp_train_nnse, nee_reco_train_nnse, et_train_nnse, lai_train_nnse, gpp_test_nnse, nee_reco_test_nnse, et_test_nnse, lai_test_nnse

def compute_test_nnse(output_matrix, target_matrix,  test_sel, reco=False):
    gpp_test_nnse = compute_nnse(target_matrix[test_sel, 0], output_matrix[test_sel, 1], target_matrix[test_sel, 1])
    if reco:
        nee_reco_test_nnse = compute_nnse(target_matrix[test_sel, 8], output_matrix[test_sel, 1] + output_matrix[test_sel, 21], target_matrix[test_sel, 9])     
    else:
        nee_reco_test_nnse = compute_nnse(target_matrix[test_sel, 2], output_matrix[test_sel, 21], target_matrix[test_sel, 3])
    et_test_nnse = compute_nnse(target_matrix[test_sel, 4], output_matrix[test_sel, 2], target_matrix[test_sel, 5])
    lai_test_nnse = compute_nnse(target_matrix[test_sel, 6], output_matrix[test_sel, 0], target_matrix[test_sel, 7])
    return gpp_test_nnse + nee_reco_test_nnse + et_test_nnse + lai_test_nnse

def negative_log_sigmoid(a, b, k=1000000):
    """
    Penalty = -log(sigmoid(k*(a-b))) = log(1 + exp(-k*(a-b))) = softplus(-k*(a-b)).

    The naive ``exp(-k*(a-b))`` form overflows when |k*(a-b)| is large (common in EDC
    terms with large k), producing inf loss and killing training.
    """
    return jnn.softplus(-k * (a - b))