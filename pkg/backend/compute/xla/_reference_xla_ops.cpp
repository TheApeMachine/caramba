// Theory-layer xla_* implementations (Markov blanket, causal, Hawkes, PC, DC, …).
// Compiled into the amalgamation after xla_init + xla_math_init + projection.

#include "xla_active_inference.h"
#include "xla_causal.h"
#include "xla_dynamic_coding.h"
#include "xla_hawkes.h"
#include "xla_markov_blanket.h"
#include "xla_predictive_coding.h"

#include "activation.h"
#include "projection.h"
#include "xla_math.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

namespace {

static std::string build_ai_free_energy_mlir(int n) {
    std::string t = "tensor<" + std::to_string(n) + "xf64>";
    return
        "module @ai_fe {\n"
        "  func.func @main(%mu: " + t + ", %log_sigma: " + t + ") -> tensor<f64> {\n"
        "    %mu2 = stablehlo.multiply %mu, %mu : " + t + "\n"
        "    %exp_ls = stablehlo.exponential %log_sigma : " + t + "\n"
        "    %one = stablehlo.constant dense<1.0> : tensor<f64>\n"
        "    %one_b = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f64>) -> " + t + "\n"
        "    %s1 = stablehlo.add %mu2, %exp_ls : " + t + "\n"
        "    %s2 = stablehlo.subtract %s1, %log_sigma : " + t + "\n"
        "    %s3 = stablehlo.subtract %s2, %one_b : " + t + "\n"
        "    %zero = stablehlo.constant dense<0.0> : tensor<f64>\n"
        "    %sum = stablehlo.reduce(%s3 init: %zero) applies stablehlo.add across dimensions = [0] : (" + t + ", tensor<f64>) -> tensor<f64>\n"
        "    %half = stablehlo.constant dense<0.5> : tensor<f64>\n"
        "    %out = stablehlo.multiply %sum, %half : tensor<f64>\n"
        "    return %out : tensor<f64>\n"
        "  }\n"
        "}\n";
}

static std::string build_ai_belief_update_mlir(int n, double lr) {
    std::ostringstream lrs;
    lrs << std::setprecision(17) << std::defaultfloat << lr;
    std::string t = "tensor<" + std::to_string(n) + "xf64>";
    std::string t2 = "tensor<" + std::to_string(n * 2) + "xf64>";
    return
        "module @ai_bu {\n"
        "  func.func @main(%mu: " + t + ", %log_sigma: " + t + ", %pred_err: " + t + ") -> " + t2 + " {\n"
        "    %lr = stablehlo.constant dense<" + lrs.str() + "> : tensor<f64>\n"
        "    %lr_b = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f64>) -> " + t + "\n"
        "    %one = stablehlo.constant dense<1.0> : tensor<f64>\n"
        "    %one_b = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f64>) -> " + t + "\n"
        "    %mu_pe = stablehlo.add %mu, %pred_err : " + t + "\n"
        "    %mu_up = stablehlo.multiply %lr_b, %mu_pe : " + t + "\n"
        "    %new_mu = stablehlo.subtract %mu, %mu_up : " + t + "\n"
        "    %exp_ls = stablehlo.exponential %log_sigma : " + t + "\n"
        "    %exp_sub = stablehlo.subtract %exp_ls, %one_b : " + t + "\n"
        "    %ls_up = stablehlo.multiply %lr_b, %exp_sub : " + t + "\n"
        "    %new_ls = stablehlo.subtract %log_sigma, %ls_up : " + t + "\n"
        "    %out = \"stablehlo.concatenate\"(%new_mu, %new_ls) {dimension = 0 : i64} : (" + t + ", " + t + ") -> " + t2 + "\n"
        "    return %out : " + t2 + "\n"
        "  }\n"
        "}\n";
}

static std::string build_ai_precision_weight_mlir(int n) {
    std::string t = "tensor<" + std::to_string(n) + "xf64>";
    return
        "module @ai_pw {\n"
        "  func.func @main(%err: " + t + ", %log_prec: " + t + ") -> " + t + " {\n"
        "    %exp_prec = stablehlo.exponential %log_prec : " + t + "\n"
        "    %out = stablehlo.multiply %err, %exp_prec : " + t + "\n"
        "    return %out : " + t + "\n"
        "  }\n"
        "}\n";
}

static std::string build_ai_efe_mlir(int n, int K, double eps) {
    std::ostringstream eps_ss;
    eps_ss << std::setprecision(17) << std::defaultfloat << eps;
    std::string tQ = "tensor<" + std::to_string(n) + "x" + std::to_string(K) + "xf64>";
    std::string tOut = "tensor<" + std::to_string(K) + "xf64>";
    return
        "module @ai_efe {\n"
        "  func.func @main(%q: " + tQ + ") -> " + tOut + " {\n"
        "    %eps = stablehlo.constant dense<" + eps_ss.str() + "> : tensor<f64>\n"
        "    %eps_b = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f64>) -> " + tQ + "\n"
        "    %q_eps = stablehlo.add %q, %eps_b : " + tQ + "\n"
        "    %log_q = stablehlo.log %q_eps : " + tQ + "\n"
        "    %q_log_q = stablehlo.multiply %q, %log_q : " + tQ + "\n"
        "    %zero = stablehlo.constant dense<0.0> : tensor<f64>\n"
        "    %sum = stablehlo.reduce(%q_log_q init: %zero) applies stablehlo.add across dimensions = [0] : (" + tQ + ", tensor<f64>) -> " + tOut + "\n"
        "    %neg_one = stablehlo.constant dense<-1.0> : tensor<f64>\n"
        "    %neg_one_b = stablehlo.broadcast_in_dim %neg_one, dims = [] : (tensor<f64>) -> " + tOut + "\n"
        "    %out = stablehlo.multiply %sum, %neg_one_b : " + tOut + "\n"
        "    return %out : " + tOut + "\n"
        "  }\n"
        "}\n";
}

static std::string build_manifold_dist_mlir(int n) {
    std::string t = "tensor<" + std::to_string(n) + "xf64>";
    return
        "module @man_dist {\n"
        "  func.func @main(%a: " + t + ", %b: " + t + ") -> tensor<f64> {\n"
        "    %diff = stablehlo.subtract %a, %b : " + t + "\n"
        "    %sq = stablehlo.multiply %diff, %diff : " + t + "\n"
        "    %zero = stablehlo.constant dense<0.0> : tensor<f64>\n"
        "    %sum = stablehlo.reduce(%sq init: %zero) applies stablehlo.add across dimensions = [0] : (" + t + ", tensor<f64>) -> tensor<f64>\n"
        "    %out = stablehlo.sqrt %sum : tensor<f64>\n"
        "    return %out : tensor<f64>\n"
        "  }\n"
        "}\n";
}

static std::string build_pc_pred_err_mlir(int n, int use_prec) {
    std::string t = "tensor<" + std::to_string(n) + "xf64>";
    if (use_prec) {
        return
            "module @pc_pe {\n"
            "  func.func @main(%x: " + t + ", %mu_hat: " + t + ", %prec: " + t + ") -> " + t + " {\n"
            "    %diff = stablehlo.subtract %x, %mu_hat : " + t + "\n"
            "    %out = stablehlo.multiply %prec, %diff : " + t + "\n"
            "    return %out : " + t + "\n"
            "  }\n"
            "}\n";
    }
    return
        "module @pc_pe {\n"
        "  func.func @main(%x: " + t + ", %mu_hat: " + t + ") -> " + t + " {\n"
        "    %out = stablehlo.subtract %x, %mu_hat : " + t + "\n"
        "    return %out : " + t + "\n"
        "  }\n"
        "}\n";
}

static std::string build_pc_update_rep_mlir(int dOut, int dIn, double lr) {
    std::ostringstream lrs;
    lrs << std::setprecision(17) << std::defaultfloat << lr;
    std::string tR = "tensor<" + std::to_string(dIn) + "xf64>";
    std::string tW = "tensor<" + std::to_string(dOut) + "x" + std::to_string(dIn) + "xf64>";
    std::string tEpsL = "tensor<" + std::to_string(dOut) + "xf64>";
    std::string tEpsS = "tensor<" + std::to_string(dIn) + "xf64>";
    return
        "module @pc_upd_rep {\n"
        "  func.func @main(%r: " + tR + ", %w: " + tW + ", %epsL: " + tEpsL + ", %epsS: " + tEpsS + ") -> " + tR + " {\n"
        "    %lr = stablehlo.constant dense<" + lrs.str() + "> : tensor<f64>\n"
        "    %lr_b = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f64>) -> " + tR + "\n"
        "    %wt_eps = stablehlo.dot_general %epsL, %w, contracting_dims = [0] x [0] : (" + tEpsL + ", " + tW + ") -> " + tR + "\n"
        "    %acc = stablehlo.subtract %wt_eps, %epsS : " + tR + "\n"
        "    %upd = stablehlo.multiply %lr_b, %acc : " + tR + "\n"
        "    %out = stablehlo.add %r, %upd : " + tR + "\n"
        "    return %out : " + tR + "\n"
        "  }\n"
        "}\n";
}

static std::string build_pc_update_w_mlir(int dOut, int dIn, double lr) {
    std::ostringstream lrs;
    lrs << std::setprecision(17) << std::defaultfloat << lr;
    std::string tR = "tensor<" + std::to_string(dIn) + "xf64>";
    std::string tW = "tensor<" + std::to_string(dOut) + "x" + std::to_string(dIn) + "xf64>";
    std::string tEpsL = "tensor<" + std::to_string(dOut) + "xf64>";
    return
        "module @pc_upd_w {\n"
        "  func.func @main(%w: " + tW + ", %epsL: " + tEpsL + ", %r: " + tR + ") -> " + tW + " {\n"
        "    %lr = stablehlo.constant dense<" + lrs.str() + "> : tensor<f64>\n"
        "    %lr_b = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f64>) -> " + tEpsL + "\n"
        "    %scaled_eps = stablehlo.multiply %lr_b, %epsL : " + tEpsL + "\n"
        "    %scaled_eps_2d = stablehlo.broadcast_in_dim %scaled_eps, dims = [0] : (" + tEpsL + ") -> " + tW + "\n"
        "    %r_2d = stablehlo.broadcast_in_dim %r, dims = [1] : (" + tR + ") -> " + tW + "\n"
        "    %upd = stablehlo.multiply %scaled_eps_2d, %r_2d : " + tW + "\n"
        "    %out = stablehlo.add %w, %upd : " + tW + "\n"
        "    return %out : " + tW + "\n"
        "  }\n"
        "}\n";
}

static std::string build_hawkes_intensity_mlir(int K, int T, double t) {
    std::ostringstream ts;
    ts << std::setprecision(17) << std::defaultfloat << t;
    std::string tK = "tensor<" + std::to_string(K) + "xf64>";
    std::string tT = "tensor<" + std::to_string(T) + "xf64>";
    std::string tKT = "tensor<" + std::to_string(K) + "x" + std::to_string(T) + "xf64>";
    return
        "module @hawkes_int {\n"
        "  func.func @main(%times: " + tT + ", %alpha: " + tK + ", %beta: " + tK + ", %mu: " + tK + ") -> " + tK + " {\n"
        "    %t_val = stablehlo.constant dense<" + ts.str() + "> : tensor<f64>\n"
        "    %t_val_b = stablehlo.broadcast_in_dim %t_val, dims = [] : (tensor<f64>) -> " + tT + "\n"
        "    %dt = stablehlo.subtract %t_val_b, %times : " + tT + "\n"
        "    %zero = stablehlo.constant dense<0.0> : tensor<f64>\n"
        "    %zero_T = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f64>) -> " + tT + "\n"
        "    %is_positive = stablehlo.compare  GT, %dt, %zero_T : (" + tT + ", " + tT + ") -> tensor<" + std::to_string(T) + "xi1>\n"
        "    %dt_pos = stablehlo.select %is_positive, %dt, %zero_T : tensor<" + std::to_string(T) + "xi1>, " + tT + "\n"
        "    %dt_2d = stablehlo.broadcast_in_dim %dt_pos, dims = [1] : (" + tT + ") -> " + tKT + "\n"
        "    %beta_2d = stablehlo.broadcast_in_dim %beta, dims = [0] : (" + tK + ") -> " + tKT + "\n"
        "    %neg_one = stablehlo.constant dense<-1.0> : tensor<f64>\n"
        "    %neg_one_b = stablehlo.broadcast_in_dim %neg_one, dims = [] : (tensor<f64>) -> " + tK + "\n"
        "    %neg_beta = stablehlo.multiply %beta, %neg_one_b : " + tK + "\n"
        "    %neg_beta_2d = stablehlo.broadcast_in_dim %neg_beta, dims = [0] : (" + tK + ") -> " + tKT + "\n"
        "    %prod = stablehlo.multiply %neg_beta_2d, %dt_2d : " + tKT + "\n"
        "    %exp_prod = stablehlo.exponential %prod : " + tKT + "\n"
        "    %is_positive_2d = stablehlo.broadcast_in_dim %is_positive, dims = [1] : (tensor<" + std::to_string(T) + "xi1>) -> tensor<" + std::to_string(K) + "x" + std::to_string(T) + "xi1>\n"
        "    %zero_KT = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f64>) -> " + tKT + "\n"
        "    %exp_filtered = stablehlo.select %is_positive_2d, %exp_prod, %zero_KT : tensor<" + std::to_string(K) + "x" + std::to_string(T) + "xi1>, " + tKT + "\n"
        "    %sum = stablehlo.reduce(%exp_filtered init: %zero) applies stablehlo.add across dimensions = [1] : (" + tKT + ", tensor<f64>) -> " + tK + "\n"
        "    %a_sum = stablehlo.multiply %alpha, %sum : " + tK + "\n"
        "    %out = stablehlo.add %mu, %a_sum : " + tK + "\n"
        "    return %out : " + tK + "\n"
        "  }\n"
        "}\n";
}

static std::string build_hawkes_kernel_matrix_mlir(int T, double alpha, double beta) {
    std::ostringstream as, bs;
    as << std::setprecision(17) << std::defaultfloat << alpha;
    bs << std::setprecision(17) << std::defaultfloat << beta;
    std::string tT = "tensor<" + std::to_string(T) + "xf64>";
    std::string tTT = "tensor<" + std::to_string(T) + "x" + std::to_string(T) + "xf64>";
    return
        "module @hawkes_km {\n"
        "  func.func @main(%times: " + tT + ") -> " + tTT + " {\n"
        "    %row_times = stablehlo.broadcast_in_dim %times, dims = [0] : (" + tT + ") -> " + tTT + "\n"
        "    %col_times = stablehlo.broadcast_in_dim %times, dims = [1] : (" + tT + ") -> " + tTT + "\n"
        "    %dt = stablehlo.subtract %col_times, %row_times : " + tTT + "\n"
        "    %zero = stablehlo.constant dense<0.0> : tensor<f64>\n"
        "    %zero_TT = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f64>) -> " + tTT + "\n"
        "    %is_positive = stablehlo.compare  GT, %dt, %zero_TT : (" + tTT + ", " + tTT + ") -> tensor<" + std::to_string(T) + "x" + std::to_string(T) + "xi1>\n"
        "    %beta_val = stablehlo.constant dense<" + bs.str() + "> : tensor<f64>\n"
        "    %neg_beta = stablehlo.negate %beta_val : tensor<f64>\n"
        "    %neg_beta_TT = stablehlo.broadcast_in_dim %neg_beta, dims = [] : (tensor<f64>) -> " + tTT + "\n"
        "    %scaled_dt = stablehlo.multiply %neg_beta_TT, %dt : " + tTT + "\n"
        "    %exp_dt = stablehlo.exponential %scaled_dt : " + tTT + "\n"
        "    %alpha_val = stablehlo.constant dense<" + as.str() + "> : tensor<f64>\n"
        "    %alpha_TT = stablehlo.broadcast_in_dim %alpha_val, dims = [] : (tensor<f64>) -> " + tTT + "\n"
        "    %kernel = stablehlo.multiply %alpha_TT, %exp_dt : " + tTT + "\n"
        "    %out = stablehlo.select %is_positive, %kernel, %zero_TT : tensor<" + std::to_string(T) + "x" + std::to_string(T) + "xi1>, " + tTT + "\n"
        "    return %out : " + tTT + "\n"
        "  }\n"
        "}\n";
}

static std::string build_hawkes_log_likelihood_mlir(int T, double integral) {
    std::ostringstream is;
    is << std::setprecision(17) << std::defaultfloat << integral;
    std::string tT = "tensor<" + std::to_string(T) + "xf64>";
    return
        "module @hawkes_ll {\n"
        "  func.func @main(%intensities: " + tT + ") -> tensor<f64> {\n"
        "    %zero = stablehlo.constant dense<0.0> : tensor<f64>\n"
        "    %zero_T = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f64>) -> " + tT + "\n"
        "    %is_positive = stablehlo.compare  GT, %intensities, %zero_T : (" + tT + ", " + tT + ") -> tensor<" + std::to_string(T) + "xi1>\n"
        "    %one = stablehlo.constant dense<1.0> : tensor<f64>\n"
        "    %one_T = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f64>) -> " + tT + "\n"
        "    %safe_int = stablehlo.select %is_positive, %intensities, %one_T : tensor<" + std::to_string(T) + "xi1>, " + tT + "\n"
        "    %log_int = stablehlo.log %safe_int : " + tT + "\n"
        "    %log_filtered = stablehlo.select %is_positive, %log_int, %zero_T : tensor<" + std::to_string(T) + "xi1>, " + tT + "\n"
        "    %sum = stablehlo.reduce(%log_filtered init: %zero) applies stablehlo.add across dimensions = [0] : (" + tT + ", tensor<f64>) -> tensor<f64>\n"
        "    %integral = stablehlo.constant dense<" + is.str() + "> : tensor<f64>\n"
        "    %out = stablehlo.subtract %sum, %integral : tensor<f64>\n"
        "    return %out : tensor<f64>\n"
        "  }\n"
        "}\n";
}

static std::string build_col_mean_mlir(int T, int D) {
    std::string tD = "tensor<" + std::to_string(D) + "xf64>";
    std::string tTD = "tensor<" + std::to_string(T) + "x" + std::to_string(D) + "xf64>";
    std::ostringstream ts;
    ts << std::setprecision(17) << std::defaultfloat << (double)T;
    return
        "module @col_mean {\n"
        "  func.func @main(%x: " + tTD + ") -> " + tD + " {\n"
        "    %zero = stablehlo.constant dense<0.0> : tensor<f64>\n"
        "    %sum = stablehlo.reduce(%x init: %zero) applies stablehlo.add across dimensions = [0] : (" + tTD + ", tensor<f64>) -> " + tD + "\n"
        "    %t_val = stablehlo.constant dense<" + ts.str() + "> : tensor<f64>\n"
        "    %t_val_b = stablehlo.broadcast_in_dim %t_val, dims = [] : (tensor<f64>) -> " + tD + "\n"
        "    %out = stablehlo.divide %sum, %t_val_b : " + tD + "\n"
        "    return %out : " + tD + "\n"
        "  }\n"
        "}\n";
}

static std::string build_covariance_mlir(int T, int D) {
    std::string tD = "tensor<" + std::to_string(D) + "xf64>";
    std::string tDD = "tensor<" + std::to_string(D) + "x" + std::to_string(D) + "xf64>";
    std::string tTD = "tensor<" + std::to_string(T) + "x" + std::to_string(D) + "xf64>";
    std::ostringstream ts;
    ts << std::setprecision(17) << std::defaultfloat << (double)(T - 1);
    return
        "module @cov {\n"
        "  func.func @main(%x: " + tTD + ", %mean: " + tD + ") -> " + tDD + " {\n"
        "    %mean_b = stablehlo.broadcast_in_dim %mean, dims = [1] : (" + tD + ") -> " + tTD + "\n"
        "    %diff = stablehlo.subtract %x, %mean_b : " + tTD + "\n"
        "    %cov_sum = stablehlo.dot_general %diff, %diff, contracting_dims = [0] x [0] : (" + tTD + ", " + tTD + ") -> " + tDD + "\n"
        "    %t_val = stablehlo.constant dense<" + ts.str() + "> : tensor<f64>\n"
        "    %t_val_b = stablehlo.broadcast_in_dim %t_val, dims = [] : (tensor<f64>) -> " + tDD + "\n"
        "    %out = stablehlo.divide %cov_sum, %t_val_b : " + tDD + "\n"
        "    return %out : " + tDD + "\n"
        "  }\n"
        "}\n";
}

static std::string build_cross_cov_mlir(int T, int N, int M) {
    std::string tN = "tensor<" + std::to_string(N) + "xf64>";
    std::string tM = "tensor<" + std::to_string(M) + "xf64>";
    std::string tNM = "tensor<" + std::to_string(N) + "x" + std::to_string(M) + "xf64>";
    std::string tTN = "tensor<" + std::to_string(T) + "x" + std::to_string(N) + "xf64>";
    std::string tTM = "tensor<" + std::to_string(T) + "x" + std::to_string(M) + "xf64>";
    std::ostringstream ts;
    ts << std::setprecision(17) << std::defaultfloat << (double)(T - 1);
    return
        "module @cross_cov {\n"
        "  func.func @main(%x: " + tTN + ", %y: " + tTM + ", %mean_x: " + tN + ", %mean_y: " + tM + ") -> " + tNM + " {\n"
        "    %mean_x_b = stablehlo.broadcast_in_dim %mean_x, dims = [1] : (" + tN + ") -> " + tTN + "\n"
        "    %mean_y_b = stablehlo.broadcast_in_dim %mean_y, dims = [1] : (" + tM + ") -> " + tTM + "\n"
        "    %diff_x = stablehlo.subtract %x, %mean_x_b : " + tTN + "\n"
        "    %diff_y = stablehlo.subtract %y, %mean_y_b : " + tTM + "\n"
        "    %cov_sum = stablehlo.dot_general %diff_x, %diff_y, contracting_dims = [0] x [0] : (" + tTN + ", " + tTM + ") -> " + tNM + "\n"
        "    %t_val = stablehlo.constant dense<" + ts.str() + "> : tensor<f64>\n"
        "    %t_val_b = stablehlo.broadcast_in_dim %t_val, dims = [] : (tensor<f64>) -> " + tNM + "\n"
        "    %out = stablehlo.divide %cov_sum, %t_val_b : " + tNM + "\n"
        "    return %out : " + tNM + "\n"
        "  }\n"
        "}\n";
}

static int ref_ensure_pjrt(const char* platform) {
    if (!platform || platform[0] == '\0') {
        return -1;
    }
    if (!g_client && xla_init(platform) != 0) {
        return -1;
    }
    if (xla_math_init(platform) != 0) {
        return -1;
    }
    if (xla_projection_init(platform) != 0) {
        return -1;
    }
    return 0;
}

static int ref_ata_rmaj(const double* A, int t, int p, double* out) {
    std::vector<double> at((size_t)p * (size_t)t);
    for (int row = 0; row < t; row++) {
        for (int col = 0; col < p; col++) {
            at[(size_t)col * (size_t)t + (size_t)row] = A[row * p + col];
        }
    }
    return xla_matmul(at.data(), A, out, p, t, p);
}

static int ref_atb_rmaj(
    const double* A, int t, int ka, const double* B, int kb, double* out) {
    (void)kb;
    std::vector<double> at((size_t)ka * (size_t)t);
    for (int row = 0; row < t; row++) {
        for (int col = 0; col < ka; col++) {
            at[(size_t)col * (size_t)t + (size_t)row] = A[row * ka + col];
        }
    }
    return xla_matmul(at.data(), B, out, ka, t, kb);
}

static int ref_matmul_nn(
    const double* A, const double* B, double* C,
    int m, int k, int n) {
    return xla_matmul(A, B, C, m, k, n);
}

static int ref_matvec_nn(const double* M, const double* x, double* y, int n) {
    return xla_matmul(M, x, y, n, n, 1);
}

static int ref_fit_ols_subset(
    const double* xMat, int t, int nx,
    const double* yVec,
    const int* idx, int nidx,
    double* betaOut, int outLen) {
    (void)outLen;
    (void)t;
    int nFeat = nx + 1;
    if (nidx == 0) {
        for (int i = 0; i < nFeat; i++) {
            betaOut[i] = std::numeric_limits<double>::quiet_NaN();
        }
        return 0;
    }
    double* xSub = (double*)std::calloc((size_t)nidx * (size_t)nFeat, sizeof(double));
    double* ySub = (double*)std::calloc((size_t)nidx, sizeof(double));
    if (!xSub || !ySub) {
        std::free(xSub);
        std::free(ySub);
        return -4;
    }
    for (int s = 0; s < nidx; s++) {
        int obs = idx[s];
        xSub[s * nFeat] = 1.0;
        for (int j = 0; j < nx; j++) {
            xSub[s * nFeat + 1 + j] = xMat[obs * nx + j];
        }
        ySub[s] = yVec[obs];
    }
    double* xtx = (double*)std::calloc((size_t)nFeat * (size_t)nFeat, sizeof(double));
    double* xtxInv = (double*)std::malloc((size_t)nFeat * (size_t)nFeat * sizeof(double));
    double* xty = (double*)std::calloc((size_t)nFeat, sizeof(double));
    if (!xtx || !xtxInv || !xty) {
        std::free(xSub);
        std::free(ySub);
        std::free(xtx);
        std::free(xtxInv);
        std::free(xty);
        return -4;
    }
    if (ref_ata_rmaj(xSub, nidx, nFeat, xtx) != 0) {
        std::free(xSub);
        std::free(ySub);
        std::free(xtx);
        std::free(xtxInv);
        std::free(xty);
        return -5;
    }
    if (xla_spd_inverse(xtx, nFeat, 1e-10, xtxInv) != 0) {
        std::free(xSub);
        std::free(ySub);
        std::free(xtx);
        std::free(xtxInv);
        std::free(xty);
        return -5;
    }
    for (int pIdx = 0; pIdx < nFeat; pIdx++) {
        xty[pIdx] = 0.0;
    }
    for (int r = 0; r < nidx; r++) {
        for (int pIdx = 0; pIdx < nFeat; pIdx++) {
            xty[pIdx] += xSub[r * nFeat + pIdx] * ySub[r];
        }
    }
    if (ref_matvec_nn(xtxInv, xty, betaOut, nFeat) != 0) {
        std::free(xSub);
        std::free(ySub);
        std::free(xtx);
        std::free(xtxInv);
        std::free(xty);
        return -5;
    }
    std::free(xSub);
    std::free(ySub);
    std::free(xtx);
    std::free(xtxInv);
    std::free(xty);
    return 0;
}

static int ref_dag_is_acyclic(const double* adj, int n) {
    int* indeg = (int*)std::calloc((size_t)n, sizeof(int));
    if (!indeg) {
        return -1;
    }
    for (int child = 0; child < n; child++) {
        for (int parent = 0; parent < n; parent++) {
            if (adj[child * n + parent] != 0.0) {
                indeg[child]++;
            }
        }
    }
    int* q = (int*)std::malloc((size_t)n * sizeof(int));
    if (!q) {
        std::free(indeg);
        return -1;
    }
    int qh = 0, qt = 0;
    for (int i = 0; i < n; i++) {
        if (indeg[i] == 0) {
            q[qt++] = i;
        }
    }
    int ord = 0;
    while (qh < qt) {
        int u = q[qh++];
        ord++;
        for (int child = 0; child < n; child++) {
            if (adj[child * n + u] == 0.0) {
                continue;
            }
            indeg[child]--;
            if (indeg[child] == 0) {
                q[qt++] = child;
            }
        }
    }
    std::free(indeg);
    std::free(q);
    return ord == n ? 0 : -2;
}

static int ref_discretize(const double* data, int n, int nBins, std::vector<int>& bins) {
    if (!data || nBins <= 0 || n <= 0) {
        return -1;
    }
    bins.resize((size_t)n);
    std::vector<double> sorted((size_t)n);
    for (int i = 0; i < n; i++) {
        sorted[(size_t)i] = data[i];
    }
    std::sort(sorted.begin(), sorted.end());
    std::vector<double> boundaries;
    if (nBins > 1) {
        boundaries.resize((size_t)(nBins - 1));
    }
    for (int binIdx = 1; binIdx < nBins; binIdx++) {
        double quantilePos = (double)binIdx / (double)nBins * (double)n;
        int intPos = (int)quantilePos;
        if (intPos >= n) {
            intPos = n - 1;
        }
        boundaries[(size_t)binIdx - 1] = sorted[(size_t)intPos];
    }
    for (int obsIdx = 0; obsIdx < n; obsIdx++) {
        double val = data[obsIdx];
        int b = 0;
        while (b < nBins - 1 && val >= boundaries[(size_t)b]) {
            b++;
        }
        bins[(size_t)obsIdx] = b;
    }
    return 0;
}

static int g_ai_inited;
static int g_causal_inited;
static int g_hawkes_inited;
static int g_mb_inited;
static int g_pc_inited;
static int g_dc_inited;

} // namespace

extern "C" {

int xla_ai_init(const char* platform) {
    if (ref_ensure_pjrt(platform) != 0) {
        return -1;
    }
    g_ai_inited = 1;
    return 0;
}

void xla_ai_shutdown(void) {
    g_ai_inited = 0;
}

int xla_ai_free_energy(const double* mu, const double* log_sigma, double* out, int n) {
    if (!g_ai_inited) return -3;
    if (!mu || !log_sigma || !out || n <= 0) return -1;
    
    std::string key = "ai_fe_" + std::to_string(n);
    auto* exec = xla_math_compile_module(key, build_ai_free_energy_mlir(n));
    if (!exec) return -5;
    const double* ins[2] = {mu, log_sigma};
    size_t sizes[2] = {(size_t)n * sizeof(double), (size_t)n * sizeof(double)};
    return xla_math_run_exec(exec, ins, 2, sizes, out, sizeof(double));
}

int xla_ai_belief_update(
    const double* mu, const double* log_sigma,
    const double* pred_err, double lr,
    double* out, int n) {
    if (!g_ai_inited) return -3;
    if (!mu || !log_sigma || !pred_err || !out || n <= 0) return -1;
    
    std::ostringstream lrs;
    lrs << std::setprecision(17) << std::defaultfloat << lr;
    std::string key = "ai_bu_" + std::to_string(n) + "_" + lrs.str();
    auto* exec = xla_math_compile_module(key, build_ai_belief_update_mlir(n, lr));
    if (!exec) return -5;
    const double* ins[3] = {mu, log_sigma, pred_err};
    size_t sizes[3] = {(size_t)n * sizeof(double), (size_t)n * sizeof(double), (size_t)n * sizeof(double)};
    return xla_math_run_exec(exec, ins, 3, sizes, out, (size_t)n * 2 * sizeof(double));
}

int xla_ai_precision_weight(
    const double* err, const double* log_prec, double* out, int n) {
    if (!g_ai_inited) return -3;
    if (!err || !log_prec || !out || n <= 0) return -1;
    
    std::string key = "ai_pw_" + std::to_string(n);
    auto* exec = xla_math_compile_module(key, build_ai_precision_weight_mlir(n));
    if (!exec) return -5;
    const double* ins[2] = {err, log_prec};
    size_t sizes[2] = {(size_t)n * sizeof(double), (size_t)n * sizeof(double)};
    return xla_math_run_exec(exec, ins, 2, sizes, out, (size_t)n * sizeof(double));
}

int xla_ai_expected_free_energy(
    const double* q_outcomes, double* out, int n, int K, double eps) {
    if (!g_ai_inited) return -3;
    if (!q_outcomes || !out || n <= 0 || K <= 0) return -1;
    if (!(eps > 0.0) || !std::isfinite(eps)) return -1;
    
    std::ostringstream eps_ss;
    eps_ss << std::setprecision(17) << std::defaultfloat << eps;
    std::string key = "ai_efe_" + std::to_string(n) + "_" + std::to_string(K) + "_" + eps_ss.str();
    auto* exec = xla_math_compile_module(key, build_ai_efe_mlir(n, K, eps));
    if (!exec) return -5;
    const double* ins[1] = {q_outcomes};
    size_t sizes[1] = {(size_t)n * (size_t)K * sizeof(double)};
    return xla_math_run_exec(exec, ins, 1, sizes, out, (size_t)K * sizeof(double));
}

int xla_causal_init(const char* platform) {
    if (ref_ensure_pjrt(platform) != 0) {
        return -1;
    }
    g_causal_inited = 1;
    return 0;
}

void xla_causal_shutdown(void) {
    g_causal_inited = 0;
}

int xla_causal_do_calculus(
    const double* cov, const double* mask, const double* values,
    double* out, int N) {
    if (!g_causal_inited) {
        return -3;
    }
    if (!cov || !mask || !values || !out || N <= 0) {
        return -1;
    }

    double* adjMean = (double*)std::calloc((size_t)N, sizeof(double));
    double* adjCov = (double*)std::malloc((size_t)N * (size_t)N * sizeof(double));
    int* intervened = (int*)std::malloc((size_t)N * sizeof(int));
    int* freev = (int*)std::malloc((size_t)N * sizeof(int));
    if (!adjMean || !adjCov || !intervened || !freev) {
        std::free(adjMean);
        std::free(adjCov);
        std::free(intervened);
        std::free(freev);
        return -4;
    }
    std::memcpy(adjCov, cov, (size_t)N * (size_t)N * sizeof(double));

    int ni = 0, nf = 0;
    for (int idx = 0; idx < N; idx++) {
        if (mask[idx] != 0.0) {
            intervened[ni++] = idx;
        } else {
            freev[nf++] = idx;
        }
    }

    for (int k = 0; k < ni; k++) {
        int ii = intervened[k];
        adjMean[ii] = values[ii];
    }

    if (ni > 0 && nf > 0) {
        double* sigII = (double*)std::calloc((size_t)ni * (size_t)ni, sizeof(double));
        double* sigFI = (double*)std::calloc((size_t)nf * (size_t)ni, sizeof(double));
        double* sigFF = (double*)std::calloc((size_t)nf * (size_t)nf, sizeof(double));
        double* sigIF = (double*)std::calloc((size_t)ni * (size_t)nf, sizeof(double));
        double* invII = (double*)std::malloc((size_t)ni * (size_t)ni * sizeof(double));
        double* xIntV = (double*)std::malloc((size_t)ni * sizeof(double));
        double* invX = (double*)std::malloc((size_t)ni * sizeof(double));
        double* delta = (double*)std::malloc((size_t)nf * sizeof(double));
        double* invIISigIF = (double*)std::malloc((size_t)ni * (size_t)nf * sizeof(double));
        double* correction = (double*)std::malloc((size_t)nf * (size_t)nf * sizeof(double));
        if (!sigII || !sigFI || !sigFF || !sigIF || !invII ||
            !xIntV || !invX || !delta || !invIISigIF || !correction) {
            std::free(sigII);
            std::free(sigFI);
            std::free(sigFF);
            std::free(sigIF);
            std::free(invII);
            std::free(xIntV);
            std::free(invX);
            std::free(delta);
            std::free(invIISigIF);
            std::free(correction);
            std::free(adjMean);
            std::free(adjCov);
            std::free(intervened);
            std::free(freev);
            return -4;
        }

        for (int r = 0; r < ni; r++) {
            for (int c = 0; c < ni; c++) {
                sigII[r * ni + c] = cov[intervened[r] * N + intervened[c]];
            }
        }
        for (int r = 0; r < nf; r++) {
            for (int c = 0; c < ni; c++) {
                sigFI[r * ni + c] = cov[freev[r] * N + intervened[c]];
            }
        }
        for (int r = 0; r < nf; r++) {
            for (int c = 0; c < nf; c++) {
                sigFF[r * nf + c] = cov[freev[r] * N + freev[c]];
            }
        }
        for (int r = 0; r < ni; r++) {
            for (int c = 0; c < nf; c++) {
                sigIF[r * nf + c] = cov[intervened[r] * N + freev[c]];
            }
        }

        if (xla_spd_inverse(sigII, ni, 1e-10, invII) != 0) {
            std::free(sigII);
            std::free(sigFI);
            std::free(sigFF);
            std::free(sigIF);
            std::free(invII);
            std::free(xIntV);
            std::free(invX);
            std::free(delta);
            std::free(invIISigIF);
            std::free(correction);
            std::free(adjMean);
            std::free(adjCov);
            std::free(intervened);
            std::free(freev);
            return -5;
        }

        for (int k = 0; k < ni; k++) {
            xIntV[k] = values[intervened[k]];
        }
        if (ref_matvec_nn(invII, xIntV, invX, ni) != 0) {
            std::free(sigII);
            std::free(sigFI);
            std::free(sigFF);
            std::free(sigIF);
            std::free(invII);
            std::free(xIntV);
            std::free(invX);
            std::free(delta);
            std::free(invIISigIF);
            std::free(correction);
            std::free(adjMean);
            std::free(adjCov);
            std::free(intervened);
            std::free(freev);
            return -5;
        }

        if (ref_matmul_nn(sigFI, invX, delta, nf, ni, 1) != 0) {
            std::free(sigII);
            std::free(sigFI);
            std::free(sigFF);
            std::free(sigIF);
            std::free(invII);
            std::free(xIntV);
            std::free(invX);
            std::free(delta);
            std::free(invIISigIF);
            std::free(correction);
            std::free(adjMean);
            std::free(adjCov);
            std::free(intervened);
            std::free(freev);
            return -5;
        }
        for (int k = 0; k < nf; k++) {
            adjMean[freev[k]] = delta[k];
        }

        if (ref_matmul_nn(invII, sigIF, invIISigIF, ni, ni, nf) != 0 ||
            ref_matmul_nn(sigFI, invIISigIF, correction, nf, ni, nf) != 0) {
            std::free(sigII);
            std::free(sigFI);
            std::free(sigFF);
            std::free(sigIF);
            std::free(invII);
            std::free(xIntV);
            std::free(invX);
            std::free(delta);
            std::free(invIISigIF);
            std::free(correction);
            std::free(adjMean);
            std::free(adjCov);
            std::free(intervened);
            std::free(freev);
            return -5;
        }
        for (int r = 0; r < nf; r++) {
            for (int c = 0; c < nf; c++) {
                adjCov[freev[r] * N + freev[c]] =
                    sigFF[r * nf + c] - correction[r * nf + c];
            }
        }

        std::free(sigII);
        std::free(sigFI);
        std::free(sigFF);
        std::free(sigIF);
        std::free(invII);
        std::free(xIntV);
        std::free(invX);
        std::free(delta);
        std::free(invIISigIF);
        std::free(correction);
    }

    for (int k = 0; k < ni; k++) {
        int ii = intervened[k];
        for (int j = 0; j < N; j++) {
            adjCov[ii * N + j] = 0.0;
            adjCov[j * N + ii] = 0.0;
        }
    }

    for (int i = 0; i < N; i++) {
        out[i] = adjMean[i];
    }
    for (int i = 0; i < N * N; i++) {
        out[N + i] = adjCov[i];
    }

    std::free(adjMean);
    std::free(adjCov);
    std::free(intervened);
    std::free(freev);
    return 0;
}

int xla_causal_backdoor(
    const double* Y, const double* X, const double* Z,
    double* effect,
    int T, int ny, int nx, int nz) {
    if (!g_causal_inited) {
        return -3;
    }
    if (!Y || !X || !effect || T <= 0 || ny <= 0 || nx <= 0 || nz < 0) {
        return -1;
    }
    if (nz > 0 && !Z) {
        return -1;
    }

    int p = 1 + nx + nz;
    double* design = (double*)std::calloc((size_t)T * (size_t)p, sizeof(double));
    if (!design) {
        return -4;
    }
    for (int row = 0; row < T; row++) {
        design[row * p] = 1.0;
        for (int col = 0; col < nx; col++) {
            design[row * p + 1 + col] = X[row * nx + col];
        }
        for (int col = 0; col < nz; col++) {
            design[row * p + 1 + nx + col] = Z[row * nz + col];
        }
    }

    double* wtw = (double*)std::calloc((size_t)p * (size_t)p, sizeof(double));
    if (!wtw) {
        std::free(design);
        return -4;
    }
    if (ref_ata_rmaj(design, T, p, wtw) != 0) {
        std::free(design);
        std::free(wtw);
        return -5;
    }

    double* wtwInv = (double*)std::malloc((size_t)p * (size_t)p * sizeof(double));
    if (!wtwInv || xla_spd_inverse(wtw, p, 1e-10, wtwInv) != 0) {
        std::free(design);
        std::free(wtw);
        std::free(wtwInv);
        return -5;
    }

    double* wty = (double*)std::calloc((size_t)p, sizeof(double));
    double* beta = (double*)std::calloc((size_t)p, sizeof(double));
    if (!wty || !beta) {
        std::free(design);
        std::free(wtw);
        std::free(wtwInv);
        std::free(wty);
        std::free(beta);
        return -4;
    }

    for (int yDim = 0; yDim < ny; yDim++) {
        for (int pIdx = 0; pIdx < p; pIdx++) {
            wty[pIdx] = 0.0;
        }
        for (int r = 0; r < T; r++) {
            double yv = Y[r * ny + yDim];
            for (int pIdx = 0; pIdx < p; pIdx++) {
                wty[pIdx] += design[r * p + pIdx] * yv;
            }
        }
        if (ref_matvec_nn(wtwInv, wty, beta, p) != 0) {
            std::free(design);
            std::free(wtw);
            std::free(wtwInv);
            std::free(wty);
            std::free(beta);
            return -5;
        }
        double eff = 0.0;
        for (int xDim = 0; xDim < nx; xDim++) {
            eff += std::fabs(beta[1 + xDim]);
        }
        effect[yDim] = eff / (double)nx;
    }

    std::free(design);
    std::free(wtw);
    std::free(wtwInv);
    std::free(wty);
    std::free(beta);
    return 0;
}

int xla_causal_frontdoor(
    const double* X, const double* M, const double* Y,
    double* effect,
    int T, int nx, int nm) {
    if (!g_causal_inited) {
        return -3;
    }
    if (!X || !M || !Y || !effect || T <= 0 || nx <= 0 || nm <= 0) {
        return -1;
    }
    std::vector<int> xBins;
    std::vector<int> mBins;
    if (ref_discretize(X, T, nx, xBins) != 0 ||
        ref_discretize(M, T, nm, mBins) != 0) {
        return -1;
    }

    std::vector<double> pX((size_t)nx, 0.0);
    for (int obsIdx = 0; obsIdx < T; obsIdx++) {
        pX[(size_t)xBins[(size_t)obsIdx]] += 1.0;
    }
    for (int i = 0; i < nx; i++) {
        pX[(size_t)i] /= (double)T;
    }

    std::vector<double> pMGivenX((size_t)nm * (size_t)nx, 0.0);
    std::vector<double> countX((size_t)nx, 0.0);
    for (int obsIdx = 0; obsIdx < T; obsIdx++) {
        int xBin = xBins[(size_t)obsIdx];
        int mBin = mBins[(size_t)obsIdx];
        pMGivenX[(size_t)mBin * (size_t)nx + (size_t)xBin] += 1.0;
        countX[(size_t)xBin] += 1.0;
    }
    for (int xBin = 0; xBin < nx; xBin++) {
        for (int mBin = 0; mBin < nm; mBin++) {
            if (countX[(size_t)xBin] > 0.0) {
                pMGivenX[(size_t)mBin * (size_t)nx + (size_t)xBin] /=
                    countX[(size_t)xBin];
            }
        }
    }

    std::vector<double> eYGivenXM((size_t)nx * (size_t)nm, 0.0);
    std::vector<double> countXM((size_t)nx * (size_t)nm, 0.0);
    for (int obsIdx = 0; obsIdx < T; obsIdx++) {
        int xBin = xBins[(size_t)obsIdx];
        int mBin = mBins[(size_t)obsIdx];
        size_t idxCell = (size_t)xBin * (size_t)nm + (size_t)mBin;
        eYGivenXM[idxCell] += Y[obsIdx];
        countXM[idxCell] += 1.0;
    }
    for (int xBin = 0; xBin < nx; xBin++) {
        for (int mBin = 0; mBin < nm; mBin++) {
            size_t idxCell = (size_t)xBin * (size_t)nm + (size_t)mBin;
            if (countXM[idxCell] > 0.0) {
                eYGivenXM[idxCell] /= countXM[idxCell];
            } else {
                eYGivenXM[idxCell] = std::numeric_limits<double>::quiet_NaN();
            }
        }
    }

    for (int xBin = 0; xBin < nx; xBin++) {
        double acc = 0.0;
        for (int mBin = 0; mBin < nm; mBin++) {
            double innerSum = 0.0;
            for (int xPrimeBin = 0; xPrimeBin < nx; xPrimeBin++) {
                double cellMean =
                    eYGivenXM[(size_t)xPrimeBin * (size_t)nm + (size_t)mBin];
                if (std::isnan(cellMean)) {
                    continue;
                }
                innerSum += cellMean * pX[(size_t)xPrimeBin];
            }
            acc += pMGivenX[(size_t)mBin * (size_t)nx + (size_t)xBin] * innerSum;
        }
        effect[xBin] = acc;
    }
    return 0;
}

int xla_causal_counterfactual(
    const double* X_obs, const double* Y_obs,
    const double* beta, const double* X_cf,
    double* Y_cf,
    int N, int N_cf) {
    if (!g_causal_inited) {
        return -3;
    }
    if (!X_obs || !Y_obs || !beta || !X_cf || !Y_cf || N <= 0 || N_cf <= 0) {
        return -1;
    }
    for (int i = 0; i < N; i++) {
        double eps = Y_obs[i] - beta[i] * X_obs[i];
        for (int j = 0; j < N_cf; j++) {
            Y_cf[i * N_cf + j] = beta[i] * X_cf[j] + eps;
        }
    }
    return 0;
}

int xla_causal_iv(
    const double* Z, const double* X, const double* Y,
    double* beta_iv,
    int T, int nz, int nx, int ny) {
    if (!g_causal_inited) {
        return -3;
    }
    if (!Z || !X || !Y || !beta_iv || T <= 0 || nz <= 0 || nx <= 0 || ny <= 0) {
        return -1;
    }

    double* ztZ = (double*)std::calloc((size_t)nz * (size_t)nz, sizeof(double));
    double* ztZInv = (double*)std::malloc((size_t)nz * (size_t)nz * sizeof(double));
    double* ztX = (double*)std::calloc((size_t)nz * (size_t)nx, sizeof(double));
    double* proj = (double*)std::malloc((size_t)nz * (size_t)nx * sizeof(double));
    double* xHat = (double*)std::malloc((size_t)T * (size_t)nx * sizeof(double));
    if (!ztZ || !ztZInv || !ztX || !proj || !xHat) {
        std::free(ztZ);
        std::free(ztZInv);
        std::free(ztX);
        std::free(proj);
        std::free(xHat);
        return -4;
    }

    if (ref_ata_rmaj(Z, T, nz, ztZ) != 0) {
        std::free(ztZ);
        std::free(ztZInv);
        std::free(ztX);
        std::free(proj);
        std::free(xHat);
        return -5;
    }
    if (xla_spd_inverse(ztZ, nz, 1e-10, ztZInv) != 0) {
        std::free(ztZ);
        std::free(ztZInv);
        std::free(ztX);
        std::free(proj);
        std::free(xHat);
        return -5;
    }
    if (ref_atb_rmaj(Z, T, nz, X, nx, ztX) != 0 ||
        ref_matmul_nn(ztZInv, ztX, proj, nz, nz, nx) != 0 ||
        ref_matmul_nn(Z, proj, xHat, T, nz, nx) != 0) {
        std::free(ztZ);
        std::free(ztZInv);
        std::free(ztX);
        std::free(proj);
        std::free(xHat);
        return -5;
    }
    double* xhTxh = (double*)std::calloc((size_t)nx * (size_t)nx, sizeof(double));
    double* xhTxhInv = (double*)std::malloc((size_t)nx * (size_t)nx * sizeof(double));
    double* xhTy = (double*)std::calloc((size_t)nx * (size_t)ny, sizeof(double));
    double* beta = (double*)std::malloc((size_t)nx * (size_t)ny * sizeof(double));
    if (!xhTxh || !xhTxhInv || !xhTy || !beta) {
        std::free(ztZ);
        std::free(ztZInv);
        std::free(ztX);
        std::free(proj);
        std::free(xHat);
        std::free(xhTxh);
        std::free(xhTxhInv);
        std::free(xhTy);
        std::free(beta);
        return -4;
    }
    if (ref_ata_rmaj(xHat, T, nx, xhTxh) != 0) {
        std::free(ztZ);
        std::free(ztZInv);
        std::free(ztX);
        std::free(proj);
        std::free(xHat);
        std::free(xhTxh);
        std::free(xhTxhInv);
        std::free(xhTy);
        std::free(beta);
        return -5;
    }
    if (xla_spd_inverse(xhTxh, nx, 1e-10, xhTxhInv) != 0) {
        std::free(ztZ);
        std::free(ztZInv);
        std::free(ztX);
        std::free(proj);
        std::free(xHat);
        std::free(xhTxh);
        std::free(xhTxhInv);
        std::free(xhTy);
        std::free(beta);
        return -5;
    }
    if (ref_atb_rmaj(xHat, T, nx, Y, ny, xhTy) != 0 ||
        ref_matmul_nn(xhTxhInv, xhTy, beta, nx, nx, ny) != 0) {
        std::free(ztZ);
        std::free(ztZInv);
        std::free(ztX);
        std::free(proj);
        std::free(xHat);
        std::free(xhTxh);
        std::free(xhTxhInv);
        std::free(xhTy);
        std::free(beta);
        return -5;
    }
    for (int i = 0; i < nx * ny; i++) {
        beta_iv[i] = beta[i];
    }

    std::free(ztZ);
    std::free(ztZInv);
    std::free(ztX);
    std::free(proj);
    std::free(xHat);
    std::free(xhTxh);
    std::free(xhTxhInv);
    std::free(xhTy);
    std::free(beta);
    return 0;
}

int xla_causal_cate(
    const double* X, const double* treatment, const double* Y,
    double* cate,
    int T, int nx) {
    if (!g_causal_inited) {
        return -3;
    }
    if (!X || !treatment || !Y || !cate || T <= 0 || nx <= 0) {
        return -1;
    }

    double* xd = (double*)std::malloc((size_t)T * (size_t)nx * sizeof(double));
    double* yd = (double*)std::malloc((size_t)T * sizeof(double));
    if (!xd || !yd) {
        std::free(xd);
        std::free(yd);
        return -4;
    }
    std::memcpy(xd, X, (size_t)T * (size_t)nx * sizeof(double));
    std::memcpy(yd, Y, (size_t)T * sizeof(double));

    int* treatIdx = (int*)std::malloc((size_t)T * sizeof(int));
    int* ctrlIdx = (int*)std::malloc((size_t)T * sizeof(int));
    if (!treatIdx || !ctrlIdx) {
        std::free(xd);
        std::free(yd);
        std::free(treatIdx);
        std::free(ctrlIdx);
        return -4;
    }
    int nt = 0, nc = 0;
    for (int i = 0; i < T; i++) {
        if (treatment[i] >= 0.5) {
            treatIdx[nt++] = i;
        } else {
            ctrlIdx[nc++] = i;
        }
    }

    double* b1 = (double*)std::malloc((size_t)(nx + 1) * sizeof(double));
    double* b0 = (double*)std::malloc((size_t)(nx + 1) * sizeof(double));
    if (!b1 || !b0) {
        std::free(xd);
        std::free(yd);
        std::free(treatIdx);
        std::free(ctrlIdx);
        std::free(b1);
        std::free(b0);
        return -4;
    }

    if (nt == 0 || nc == 0) {
        for (int i = 0; i < T; i++) {
            cate[i] = std::numeric_limits<double>::quiet_NaN();
        }
        std::free(xd);
        std::free(yd);
        std::free(treatIdx);
        std::free(ctrlIdx);
        std::free(b1);
        std::free(b0);
        return 0;
    }

    if (ref_fit_ols_subset(xd, T, nx, yd, treatIdx, nt, b1, nx + 1) != 0 ||
        ref_fit_ols_subset(xd, T, nx, yd, ctrlIdx, nc, b0, nx + 1) != 0) {
        std::free(xd);
        std::free(yd);
        std::free(treatIdx);
        std::free(ctrlIdx);
        std::free(b1);
        std::free(b0);
        return -5;
    }

    for (int obs = 0; obs < T; obs++) {
        double p1 = b1[0];
        double p0 = b0[0];
        for (int j = 0; j < nx; j++) {
            double xv = xd[obs * nx + j];
            p1 += b1[1 + j] * xv;
            p0 += b0[1 + j] * xv;
        }
        cate[obs] = p1 - p0;
    }

    std::free(xd);
    std::free(yd);
    std::free(treatIdx);
    std::free(ctrlIdx);
    std::free(b1);
    std::free(b0);
    return 0;
}

int xla_causal_dag_markov(
    const double* X, const double* adj,
    double* log_prob,
    int T, int N) {
    if (!g_causal_inited) {
        return -3;
    }
    if (!X || !adj || !log_prob || T <= 0 || N <= 0) {
        return -1;
    }
    if (ref_dag_is_acyclic(adj, N) != 0) {
        return -2;
    }

    double* nodeBetaFlat =
        (double*)std::malloc((size_t)N * (size_t)(N + 1) * sizeof(double));
    int* nodeBetaLen = (int*)std::calloc((size_t)N, sizeof(int));
    double* nodeSigma2 = (double*)std::calloc((size_t)N, sizeof(double));
    int** parents = (int**)std::calloc((size_t)N, sizeof(int*));
    if (!nodeBetaFlat || !nodeBetaLen || !nodeSigma2 || !parents) {
        std::free(nodeBetaFlat);
        std::free(nodeBetaLen);
        std::free(nodeSigma2);
        std::free(parents);
        return -4;
    }

    double* workX = (double*)std::malloc((size_t)T * (size_t)N * sizeof(double));
    double* nodeVals = (double*)std::malloc((size_t)T * sizeof(double));
    if (!workX || !nodeVals) {
        std::free(nodeBetaFlat);
        std::free(nodeBetaLen);
        std::free(nodeSigma2);
        for (int i = 0; i < N; i++) {
            std::free(parents[i]);
        }
        std::free(parents);
        std::free(workX);
        std::free(nodeVals);
        return -4;
    }
    std::memcpy(workX, X, (size_t)T * (size_t)N * sizeof(double));

    for (int nodeIdx = 0; nodeIdx < N; nodeIdx++) {
        int np = 0;
        for (int j = 0; j < N; j++) {
            if (adj[nodeIdx * N + j] != 0.0) {
                np++;
            }
        }
        parents[nodeIdx] = (int*)std::malloc((size_t)np * sizeof(int));
        if (!parents[nodeIdx]) {
            for (int k = 0; k <= nodeIdx; k++) {
                std::free(parents[k]);
            }
            std::free(parents);
            std::free(nodeBetaFlat);
            std::free(nodeBetaLen);
            std::free(nodeSigma2);
            std::free(workX);
            std::free(nodeVals);
            return -4;
        }
        int p = 0;
        for (int j = 0; j < N; j++) {
            if (adj[nodeIdx * N + j] != 0.0) {
                parents[nodeIdx][p++] = j;
            }
        }

        for (int obs = 0; obs < T; obs++) {
            nodeVals[obs] = workX[obs * N + nodeIdx];
        }

        double* betaDest = nodeBetaFlat + nodeIdx * (N + 1);

        if (np == 0) {
            double mean = 0.0;
            for (int o = 0; o < T; o++) {
                mean += nodeVals[o];
            }
            mean /= (double)T;
            double var = 0.0;
            for (int o = 0; o < T; o++) {
                double d = nodeVals[o] - mean;
                var += d * d;
            }
            var /= (double)T;
            if (var < 1e-10) {
                var = 1e-10;
            }
            betaDest[0] = mean;
            nodeBetaLen[nodeIdx] = 1;
            nodeSigma2[nodeIdx] = var;
        } else if (T <= np) {
            nodeBetaLen[nodeIdx] = np + 1;
            for (int k = 0; k < np + 1; k++) {
                betaDest[k] = 0.0;
            }
            nodeSigma2[nodeIdx] = 1e-10;
        } else {
            double* pMat =
                (double*)std::malloc((size_t)T * (size_t)np * sizeof(double));
            if (!pMat) {
                for (int k = 0; k < N; k++) {
                    std::free(parents[k]);
                }
                std::free(parents);
                std::free(nodeBetaFlat);
                std::free(nodeBetaLen);
                std::free(nodeSigma2);
                std::free(workX);
                std::free(nodeVals);
                return -4;
            }
            for (int o = 0; o < T; o++) {
                for (int pj = 0; pj < np; pj++) {
                    int pjnode = parents[nodeIdx][pj];
                    pMat[o * np + pj] = workX[o * N + pjnode];
                }
            }
            int* allIdx = (int*)std::malloc((size_t)T * sizeof(int));
            for (int o = 0; o < T; o++) {
                allIdx[o] = o;
            }
            if (ref_fit_ols_subset(
                    pMat, T, np, nodeVals, allIdx, T,
                    betaDest, np + 1) != 0) {
                std::free(pMat);
                std::free(allIdx);
                for (int k = 0; k < N; k++) {
                    std::free(parents[k]);
                }
                std::free(parents);
                std::free(nodeBetaFlat);
                std::free(nodeBetaLen);
                std::free(nodeSigma2);
                std::free(workX);
                std::free(nodeVals);
                return -5;
            }
            nodeBetaLen[nodeIdx] = np + 1;
            double rss = 0.0;
            for (int o = 0; o < T; o++) {
                double pred = betaDest[0];
                for (int pj = 0; pj < np; pj++) {
                    pred += betaDest[1 + pj] * pMat[o * np + pj];
                }
                double res = nodeVals[o] - pred;
                rss += res * res;
            }
            double s2 = rss / (double)T;
            if (s2 < 1e-10) {
                s2 = 1e-10;
            }
            nodeSigma2[nodeIdx] = s2;
            std::free(pMat);
            std::free(allIdx);
        }
    }

    const double pi = 3.14159265358979323846;
    for (int obsIdx = 0; obsIdx < T; obsIdx++) {
        double logP = 0.0;
        for (int nodeIdx = 0; nodeIdx < N; nodeIdx++) {
            int np = 0;
            for (int j = 0; j < N; j++) {
                if (adj[nodeIdx * N + j] != 0.0) {
                    np++;
                }
            }
            double s2 = nodeSigma2[nodeIdx];
            double xVal = workX[obsIdx * N + nodeIdx];
            double predicted = 0.0;
            const double* betaDest = nodeBetaFlat + nodeIdx * (N + 1);
            if (np == 0) {
                predicted = betaDest[0];
            } else {
                predicted = betaDest[0];
                for (int pj = 0; pj < np; pj++) {
                    int pn = parents[nodeIdx][pj];
                    predicted += betaDest[1 + pj] * workX[obsIdx * N + pn];
                }
            }
            double diff = xVal - predicted;
            logP += -0.5 * std::log(2.0 * pi * s2) - 0.5 * diff * diff / s2;
        }
        log_prob[obsIdx] = logP;
    }

    for (int k = 0; k < N; k++) {
        std::free(parents[k]);
    }
    std::free(parents);
    std::free(nodeBetaFlat);
    std::free(nodeBetaLen);
    std::free(nodeSigma2);
    std::free(workX);
    std::free(nodeVals);
    return 0;
}

int xla_hawkes_init(const char* platform) {
    if (ref_ensure_pjrt(platform) != 0) {
        return -1;
    }
    g_hawkes_inited = 1;
    return 0;
}

void xla_hawkes_shutdown(void) {
    g_hawkes_inited = 0;
}

int xla_hawkes_intensity(
    const double* times, const double* alpha,
    const double* beta, const double* mu,
    double t,
    double* out,
    int K, int T) {
    if (!g_hawkes_inited) return -3;
    if (!times || !alpha || !beta || !mu || !out || K <= 0 || T < 0) return -1;
    
    if (T == 0) {
        std::memcpy(out, mu, (size_t)K * sizeof(double));
        return 0;
    }
    
    std::ostringstream ts;
    ts << std::setprecision(17) << std::defaultfloat << t;
    std::string key = "hawkes_int_" + std::to_string(K) + "_" + std::to_string(T) + "_" + ts.str();
    auto* exec = xla_math_compile_module(key, build_hawkes_intensity_mlir(K, T, t));
    if (!exec) return -5;
    const double* ins[4] = {times, alpha, beta, mu};
    size_t sizes[4] = {
        (size_t)T * sizeof(double),
        (size_t)K * sizeof(double),
        (size_t)K * sizeof(double),
        (size_t)K * sizeof(double)
    };
    return xla_math_run_exec(exec, ins, 4, sizes, out, (size_t)K * sizeof(double));
}

int xla_hawkes_kernel_matrix(
    const double* times,
    double alpha, double beta,
    double* out,
    int T) {
    if (!g_hawkes_inited) return -3;
    if (!times || !out || T <= 0) return -1;
    
    std::ostringstream as, bs;
    as << std::setprecision(17) << std::defaultfloat << alpha;
    bs << std::setprecision(17) << std::defaultfloat << beta;
    std::string key = "hawkes_km_" + std::to_string(T) + "_" + as.str() + "_" + bs.str();
    auto* exec = xla_math_compile_module(key, build_hawkes_kernel_matrix_mlir(T, alpha, beta));
    if (!exec) return -5;
    const double* ins[1] = {times};
    size_t sizes[1] = {(size_t)T * sizeof(double)};
    return xla_math_run_exec(exec, ins, 1, sizes, out, (size_t)T * (size_t)T * sizeof(double));
}

int xla_hawkes_log_likelihood(
    const double* intensities,
    double integral,
    double* out,
    int T) {
    if (!g_hawkes_inited) return -3;
    if (!intensities || !out || T <= 0) return -1;
    
    std::ostringstream is;
    is << std::setprecision(17) << std::defaultfloat << integral;
    std::string key = "hawkes_ll_" + std::to_string(T) + "_" + is.str();
    auto* exec = xla_math_compile_module(key, build_hawkes_log_likelihood_mlir(T, integral));
    if (!exec) return -5;
    const double* ins[1] = {intensities};
    size_t sizes[1] = {(size_t)T * sizeof(double)};
    return xla_math_run_exec(exec, ins, 1, sizes, out, sizeof(double));
}

int xla_hawkes_simulate(
    const double* mu, const double* alpha,
    const double* beta,
    double T_max, int K, int maxSteps,
    double* out) {
    if (!g_hawkes_inited) {
        return -3;
    }
    if (!mu || !alpha || !beta || !out || K <= 0 || maxSteps <= 0 || T_max <= 0.0) {
        return -1;
    }
    for (int i = 0; i < K * maxSteps; i++) {
        out[i] = -1.0;
    }
    for (int k = 0; k < K; k++) {
        uint64_t seed = (uint64_t)(k + 1) * 6364136223846793005ULL + 1442695040888963407ULL;
        double* kevents = out + k * maxSteps;
        double tev = 0.0;
        int count = 0;
        double muk = mu[k];
        double alphak = alpha[k];
        double betak = beta[k];
        while (tev < T_max && count < maxSteps) {
            double lstar = muk;
            for (int i = 0; i < count; i++) {
                lstar += alphak * std::exp(-betak * (tev - kevents[i]));
            }
            seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
            double u1 = (double)(seed >> 11) / (double)(1ULL << 53);
            if (u1 < 1e-300) {
                u1 = 1e-300;
            }
            double dt = -std::log(u1) / lstar;
            tev += dt;
            if (tev >= T_max) {
                break;
            }
            double lam = muk;
            for (int i = 0; i < count; i++) {
                lam += alphak * std::exp(-betak * (tev - kevents[i]));
            }
            seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
            double u2 = (double)(seed >> 11) / (double)(1ULL << 53);
            if (u2 <= lam / lstar) {
                kevents[count++] = tev;
            }
        }
    }
    return 0;
}

int xla_mb_init(const char* platform) {
    if (ref_ensure_pjrt(platform) != 0) {
        return -1;
    }
    g_mb_inited = 1;
    return 0;
}

int xla_mb_partition(
    const double* x, const double* masks,
    double* out,
    int N, int Ns, int Na, int Ni, int Ne) {
    if (!g_mb_inited) {
        return -3;
    }
    if (!x || !masks || !out || N <= 0) {
        return -1;
    }
    if (Ns < 0 || Na < 0 || Ni < 0 || Ne < 0) {
        return -1;
    }
    const double* smask = masks;
    const double* amask = masks + N;
    const double* imask = masks + 2 * N;
    const double* emask = masks + 3 * N;

    for (int idx = 0; idx < N; idx++) {
        int set = (smask[idx] != 0.0) + (amask[idx] != 0.0) +
                  (imask[idx] != 0.0) + (emask[idx] != 0.0);
        if (set > 1) {
            return -2;
        }
    }

    std::memset(out, 0, (size_t)(Ns + Na + Ni + Ne) * sizeof(double));
    int si = 0, ai = Ns, ii = Ns + Na, ei = Ns + Na + Ni;

    for (int idx = 0; idx < N; idx++) {
        double v = x[idx];
        if (smask[idx] != 0.0 && si < Ns) {
            out[si++] = v;
        } else if (amask[idx] != 0.0 && ai < Ns + Na) {
            out[ai++] = v;
        } else if (imask[idx] != 0.0 && ii < Ns + Na + Ni) {
            out[ii++] = v;
        } else if (emask[idx] != 0.0 && ei < Ns + Na + Ni + Ne) {
            out[ei++] = v;
        }
    }
    return 0;
}

int xla_mb_flow_internal(
    const double* x_sens, const double* W, const double* bias,
    double* out,
    int Ni, int Ns) {
    if (!g_mb_inited) {
        return -3;
    }
    if (!x_sens || !W || !bias || !out || Ni <= 0 || Ns <= 0) {
        return -1;
    }
    return xla_linear(x_sens, W, bias, out, 1, Ns, Ni, 1);
}

int xla_mb_flow_active(
    const double* x_int, const double* W, const double* bias,
    double* out,
    int Na, int Ni) {
    if (!g_mb_inited) {
        return -3;
    }
    if (!x_int || !W || !bias || !out || Na <= 0 || Ni <= 0) {
        return -1;
    }
    return xla_linear(x_int, W, bias, out, 1, Ni, Na, 1);
}

int xla_mb_mutual_information(
    const double* X, const double* Y,
    double* out,
    int T, int N, int M) {
    if (!g_mb_inited) {
        return -3;
    }
    if (!X || !Y || !out || T < 2 || N <= 0 || M <= 0) {
        return -1;
    }
    double* xm = (double*)std::calloc((size_t)N, sizeof(double));
    double* ym = (double*)std::calloc((size_t)M, sizeof(double));
    double* cx = (double*)std::calloc((size_t)N * (size_t)N, sizeof(double));
    double* cy = (double*)std::calloc((size_t)M * (size_t)M, sizeof(double));
    double* cxy = (double*)std::calloc((size_t)N * (size_t)M, sizeof(double));
    int NM = N + M;
    double* joint = (double*)std::calloc((size_t)NM * (size_t)NM, sizeof(double));
    if (!xm || !ym || !cx || !cy || !cxy || !joint) {
        std::free(xm);
        std::free(ym);
        std::free(cx);
        std::free(cy);
        std::free(cxy);
        std::free(joint);
        return -4;
    }

    std::string key_mean_x = "col_mean_" + std::to_string(T) + "_" + std::to_string(N);
    auto* exec_mean_x = xla_math_compile_module(key_mean_x, build_col_mean_mlir(T, N));
    if (!exec_mean_x) return -5;
    const double* ins_mx[1] = {X};
    size_t sizes_mx[1] = {(size_t)T * (size_t)N * sizeof(double)};
    if (xla_math_run_exec(exec_mean_x, ins_mx, 1, sizes_mx, xm, (size_t)N * sizeof(double)) != 0) return -5;

    std::string key_mean_y = "col_mean_" + std::to_string(T) + "_" + std::to_string(M);
    auto* exec_mean_y = xla_math_compile_module(key_mean_y, build_col_mean_mlir(T, M));
    if (!exec_mean_y) return -5;
    const double* ins_my[1] = {Y};
    size_t sizes_my[1] = {(size_t)T * (size_t)M * sizeof(double)};
    if (xla_math_run_exec(exec_mean_y, ins_my, 1, sizes_my, ym, (size_t)M * sizeof(double)) != 0) return -5;

    std::string key_cov_x = "cov_" + std::to_string(T) + "_" + std::to_string(N);
    auto* exec_cov_x = xla_math_compile_module(key_cov_x, build_covariance_mlir(T, N));
    if (!exec_cov_x) return -5;
    const double* ins_cx[2] = {X, xm};
    size_t sizes_cx[2] = {(size_t)T * (size_t)N * sizeof(double), (size_t)N * sizeof(double)};
    if (xla_math_run_exec(exec_cov_x, ins_cx, 2, sizes_cx, cx, (size_t)N * (size_t)N * sizeof(double)) != 0) return -5;

    std::string key_cov_y = "cov_" + std::to_string(T) + "_" + std::to_string(M);
    auto* exec_cov_y = xla_math_compile_module(key_cov_y, build_covariance_mlir(T, M));
    if (!exec_cov_y) return -5;
    const double* ins_cy[2] = {Y, ym};
    size_t sizes_cy[2] = {(size_t)T * (size_t)M * sizeof(double), (size_t)M * sizeof(double)};
    if (xla_math_run_exec(exec_cov_y, ins_cy, 2, sizes_cy, cy, (size_t)M * (size_t)M * sizeof(double)) != 0) return -5;

    std::string key_cross_cov = "cross_cov_" + std::to_string(T) + "_" + std::to_string(N) + "_" + std::to_string(M);
    auto* exec_cross_cov = xla_math_compile_module(key_cross_cov, build_cross_cov_mlir(T, N, M));
    if (!exec_cross_cov) return -5;
    const double* ins_cxy[4] = {X, Y, xm, ym};
    size_t sizes_cxy[4] = {
        (size_t)T * (size_t)N * sizeof(double),
        (size_t)T * (size_t)M * sizeof(double),
        (size_t)N * sizeof(double),
        (size_t)M * sizeof(double)
    };
    if (xla_math_run_exec(exec_cross_cov, ins_cxy, 4, sizes_cxy, cxy, (size_t)N * (size_t)M * sizeof(double)) != 0) return -5;

    for (int row = 0; row < N; row++) {
        for (int col = 0; col < N; col++) {
            joint[row * NM + col] = cx[row * N + col];
        }
    }
    for (int row = 0; row < M; row++) {
        for (int col = 0; col < M; col++) {
            joint[(N + row) * NM + (N + col)] = cy[row * M + col];
        }
    }
    for (int row = 0; row < N; row++) {
        for (int col = 0; col < M; col++) {
            joint[row * NM + (N + col)] = cxy[row * M + col];
            joint[(N + col) * NM + row] = cxy[row * M + col];
        }
    }

    double ldX;
    if (xla_spd_log_det(cx, N, 1e-10, &ldX) != 0) {
        std::free(cx);
        std::free(cy);
        std::free(cxy);
        std::free(joint);
        std::free(xm);
        std::free(ym);
        return -5;
    }
    std::free(cx);

    double ldY;
    if (xla_spd_log_det(cy, M, 1e-10, &ldY) != 0) {
        std::free(cy);
        std::free(cxy);
        std::free(joint);
        std::free(xm);
        std::free(ym);
        return -5;
    }
    std::free(cy);

    double ldJ;
    if (xla_spd_log_det(joint, NM, 1e-10, &ldJ) != 0) {
        std::free(joint);
        std::free(xm);
        std::free(ym);
        std::free(cxy);
        return -5;
    }
    std::free(joint);
    std::free(xm);
    std::free(ym);
    std::free(cxy);

    if (std::isnan(ldX) || std::isnan(ldY) || std::isnan(ldJ)) {
        return -5;
    }
    double mi = 0.5 * (ldX + ldY - ldJ);
    if (mi < 0.0) {
        mi = 0.0;
    }
    if (!std::isfinite(mi)) {
        return -5;
    }
    out[0] = mi;
    return 0;
}

int xla_mb_shutdown(void) {
    g_mb_inited = 0;
    return 0;
}

int xla_pc_init(const char* platform) {
    if (ref_ensure_pjrt(platform) != 0) {
        return -1;
    }
    g_pc_inited = 1;
    return 0;
}

void xla_pc_shutdown(void) {
    g_pc_inited = 0;
}

int xla_pc_prediction(const double* W, const double* r, double* dst, int D_out, int D_in) {
    if (!g_pc_inited) {
        return -3;
    }
    if (!W || !r || !dst || D_out <= 0 || D_in <= 0) {
        return -1;
    }
    return xla_linear(r, W, nullptr, dst, 1, D_in, D_out, 0);
}

int xla_pc_prediction_error(
    const double* x, const double* mu_hat,
    const double* prec, double* dst, int n, int use_prec) {
    if (!g_pc_inited) return -3;
    if (!x || !mu_hat || !dst || n <= 0) return -1;
    if (use_prec && !prec) return -1;
    
    std::string key = "pc_pe_" + std::to_string(n) + "_" + std::to_string(use_prec);
    auto* exec = xla_math_compile_module(key, build_pc_pred_err_mlir(n, use_prec));
    if (!exec) return -5;
    if (use_prec) {
        const double* ins[3] = {x, mu_hat, prec};
        size_t sizes[3] = {(size_t)n * sizeof(double), (size_t)n * sizeof(double), (size_t)n * sizeof(double)};
        return xla_math_run_exec(exec, ins, 3, sizes, dst, (size_t)n * sizeof(double));
    }
    const double* ins[2] = {x, mu_hat};
    size_t sizes[2] = {(size_t)n * sizeof(double), (size_t)n * sizeof(double)};
    return xla_math_run_exec(exec, ins, 2, sizes, dst, (size_t)n * sizeof(double));
}

int xla_pc_update_representation(
    const double* r, const double* W,
    const double* eps_lower, const double* eps_self,
    double lr, double* dst, int D_out, int D_in) {
    if (!g_pc_inited) return -3;
    if (!r || !W || !eps_lower || !eps_self || !dst || D_out <= 0 || D_in <= 0) return -1;
    
    std::ostringstream lrs;
    lrs << std::setprecision(17) << std::defaultfloat << lr;
    std::string key = "pc_upd_rep_" + std::to_string(D_out) + "_" + std::to_string(D_in) + "_" + lrs.str();
    auto* exec = xla_math_compile_module(key, build_pc_update_rep_mlir(D_out, D_in, lr));
    if (!exec) return -5;
    const double* ins[4] = {r, W, eps_lower, eps_self};
    size_t sizes[4] = {
        (size_t)D_in * sizeof(double),
        (size_t)D_out * (size_t)D_in * sizeof(double),
        (size_t)D_out * sizeof(double),
        (size_t)D_in * sizeof(double)
    };
    return xla_math_run_exec(exec, ins, 4, sizes, dst, (size_t)D_in * sizeof(double));
}

int xla_pc_update_weights(
    const double* W, const double* eps, const double* r,
    double lr, double* dst, int D_out, int D_in) {
    if (!g_pc_inited) return -3;
    if (!W || !eps || !r || !dst || D_out <= 0 || D_in <= 0) return -1;
    
    std::ostringstream lrs;
    lrs << std::setprecision(17) << std::defaultfloat << lr;
    std::string key = "pc_upd_w_" + std::to_string(D_out) + "_" + std::to_string(D_in) + "_" + lrs.str();
    auto* exec = xla_math_compile_module(key, build_pc_update_w_mlir(D_out, D_in, lr));
    if (!exec) return -5;
    const double* ins[3] = {W, eps, r};
    size_t sizes[3] = {
        (size_t)D_out * (size_t)D_in * sizeof(double),
        (size_t)D_out * sizeof(double),
        (size_t)D_in * sizeof(double)
    };
    return xla_math_run_exec(exec, ins, 3, sizes, dst, (size_t)D_out * (size_t)D_in * sizeof(double));
}

int xla_dc_init(const char* platform) {
    if (ref_ensure_pjrt(platform) != 0) {
        return -1;
    }
    g_dc_inited = 1;
    return 0;
}

void xla_dc_shutdown(void) {
    g_dc_inited = 0;
}

int xla_dc_project(const double* W, const double* x, double* dst, int dOut, int dIn) {
    if (!g_dc_inited) {
        return -3;
    }
    if (!W || !x || !dst || dOut <= 0 || dIn <= 0) {
        return -1;
    }
    return xla_linear(x, W, nullptr, dst, 1, dIn, dOut, 0);
}

int xla_dc_reconstruct(const double* W, const double* z, double* dst, int dIn, int dOut) {
    if (!g_dc_inited) {
        return -3;
    }
    if (!W || !z || !dst || dIn <= 0 || dOut <= 0) {
        return -1;
    }
    return xla_matmul(z, W, dst, 1, dOut, dIn);
}

int xla_dc_dynamics(
    const double* z, const double* A,
    const double* u, const double* B,
    double* dst, int d, int du) {
    if (!g_dc_inited) {
        return -3;
    }
    if (!z || !A || !u || !B || !dst || d <= 0 || du <= 0) {
        return -1;
    }
    std::vector<double> az((size_t)d);
    std::vector<double> bu((size_t)d);

    if (xla_matmul(A, z, az.data(), d, d, 1) != 0) {
        return -5;
    }
    if (xla_matmul(B, u, bu.data(), d, du, 1) != 0) {
        return -5;
    }

    return xla_add(az.data(), bu.data(), dst, d);
}

int xla_dc_manifold_distance(const double* a, const double* b, double* dist, int d) {
    if (!g_dc_inited) return -3;
    if (!a || !b || !dist || d <= 0) return -1;
    
    std::string key = "man_dist_" + std::to_string(d);
    auto* exec = xla_math_compile_module(key, build_manifold_dist_mlir(d));
    if (!exec) return -5;
    const double* ins[2] = {a, b};
    size_t sizes[2] = {(size_t)d * sizeof(double), (size_t)d * sizeof(double)};
    return xla_math_run_exec(exec, ins, 2, sizes, dist, sizeof(double));
}

} // extern "C"
