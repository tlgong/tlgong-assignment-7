from flask import Flask, render_template, request, url_for, redirect, flash
import numpy as np
import matplotlib
from scipy import stats
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

app = Flask(__name__)
app.secret_key = "your_secret_key_here"


class GlobalData:
    def __init__(self):
        self.N = None
        self.mu = None
        self.sigma2 = None
        self.beta0 = None
        self.beta1 = None
        self.S = None
        self.X = None
        self.Y = None
        self.slope = None
        self.intercept = None
        self.slopes = []
        self.intercepts = []
        self.slope_extreme = None
        self.intercept_extreme = None


data_store = GlobalData()

def generate_data(N, mu, beta0, beta1, sigma2, S):
    X = np.random.uniform(low=0, high=1, size=N)

    error = np.random.normal(loc=0, scale=np.sqrt(sigma2), size=N)
    Y = beta0 + beta1 * X + mu + error

    model = LinearRegression().fit(X.reshape(-1, 1), Y)
    slope = model.coef_[0]
    intercept = model.intercept_


    plot1_path = "static/plot1.png"
    plt.figure(figsize=(8, 6))
    plt.scatter(X, Y, color='blue', label='Data Points')
    plt.plot(X, model.predict(X.reshape(-1, 1)), color='red', label='Fitted Line')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Scatter Plot with Fitted Regression Line')
    plt.legend()
    plt.savefig(plot1_path)
    plt.close()


    slopes = []
    intercepts = []

    for _ in range(S):

        X_sim = np.random.uniform(low=0, high=1, size=N)
        error_sim = np.random.normal(loc=0, scale=np.sqrt(sigma2), size=N)
        Y_sim = beta0 + beta1 * X_sim + mu + error_sim


        sim_model = LinearRegression().fit(X_sim.reshape(-1, 1), Y_sim)
        sim_slope = sim_model.coef_[0]
        sim_intercept = sim_model.intercept_

        slopes.append(sim_slope)
        intercepts.append(sim_intercept)


    plot2_path = "static/plot2.png"
    plt.figure(figsize=(14, 6))


    plt.subplot(1, 2, 1)
    plt.hist(slopes, bins=30, color='skyblue', edgecolor='black')
    plt.axvline(slope, color='red', linestyle='dashed', linewidth=2, label=f'Observed Slope = {slope:.4f}')
    plt.axvline(beta1, color='blue', linestyle='-', linewidth=2, label=f"Hypothesized Slope (H₀): {beta1}")
    plt.xlabel('Slope')
    plt.ylabel('Frequency')
    plt.title('Histogram of Simulated Slopes')
    plt.legend()


    plt.subplot(1, 2, 2)
    plt.hist(intercepts, bins=30, color='lightgreen', edgecolor='black')
    plt.axvline(intercept, color='red', linestyle='dashed', linewidth=2, label=f'Observed Intercept = {intercept:.4f}')
    plt.axvline(beta0, color='blue', linestyle='-', linewidth=2, label=f"Hypothesized Intercept (H₀): {beta0}")
    plt.xlabel('Intercept')
    plt.ylabel('Frequency')
    plt.title('Histogram of Simulated Intercepts')
    plt.legend()

    plt.tight_layout()
    plt.savefig(plot2_path)
    plt.close()

    slope_more_extreme = np.mean(np.abs(slopes) >= np.abs(slope))
    intercept_extreme = np.mean(np.abs(intercepts) >= np.abs(intercept))


    data_store.N = N
    data_store.mu = mu
    data_store.sigma2 = sigma2
    data_store.beta0 = beta0
    data_store.beta1 = beta1
    data_store.S = S
    data_store.X = X
    data_store.Y = Y
    data_store.slope = slope
    data_store.intercept = intercept
    data_store.slopes = slopes
    data_store.intercepts = intercepts
    data_store.slope_extreme = slope_more_extreme
    data_store.intercept_extreme = intercept_extreme

    # 返回数据以供进一步分析
    return (
        X,
        Y,
        slope,
        intercept,
        plot1_path,
        plot2_path,
        slope_more_extreme,
        intercept_extreme,
        slopes,
        intercepts,
    )

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # 从表单获取用户输入

        try:
            N = int(request.form["N"])
            mu = float(request.form["mu"])
            sigma2 = float(request.form["sigma2"])
            beta0 = float(request.form["beta0"])
            beta1 = float(request.form["beta1"])
            S = int(request.form["S"])
        except (ValueError, KeyError) as e:
            flash("Invalid input. Please enter valid numbers for all fields.")
            return redirect(url_for('index'))

        (
            X,
            Y,
            slope,
            intercept,
            plot1,
            plot2,
            slope_extreme,
            intercept_extreme,
            slopes,
            intercepts,
        ) = generate_data(N, mu, beta0, beta1, sigma2, S)



        print(f"Intercept: {intercept}, Slope: {slope}")


        return render_template(
            "index.html",
            plot1=plot1,
            plot2=plot2,
            slope_extreme=slope_extreme,
            intercept_extreme=intercept_extreme,
            N=N,
            mu=mu,
            sigma2=sigma2,
            beta0=beta0,
            beta1=beta1,
            S=S,
        )
    return render_template("index.html")

@app.route("/generate", methods=["POST"])
def generate():
    # 这个路由处理数据生成，与上面的 index 路由相同
    return index()

@app.route("/hypothesis_test", methods=["POST"])
def hypothesis_test():
    try:
        print(data_store.N)
        N = data_store.N
        S = data_store.S
        slope = data_store.slope
        intercept = data_store.intercept
        slopes = data_store.slopes
        intercepts = data_store.intercepts
        beta0 = data_store.beta0
        beta1 = data_store.beta1
        print(f"Intercept: {intercept}, Slope: {slope}")
        parameter = request.form.get("parameter")
        test_type = request.form.get("test_type")

        if parameter == "slope":
            simulated_stats = np.array(slopes)
            observed_stat = slope
            hypothesized_value = beta1
        else:
            simulated_stats = np.array(intercepts)
            observed_stat = intercept
            hypothesized_value = beta0
        print(f"Test Type: {test_type}")
        if test_type == "<":
            p_value = np.mean(simulated_stats <  observed_stat)
        elif test_type == ">":
            p_value = np.mean(simulated_stats > observed_stat)
        elif test_type == "!=":
            p_left = np.mean(simulated_stats < observed_stat)
            p_right = np.mean(simulated_stats > observed_stat)
            p_value = p_left + p_right

        fun_message = None
        if p_value <= 0.0001:
            fun_message = "Extremely significant result!"
        elif p_value <= 0.01:
            fun_message = "Highly significant! "
        elif p_value <= 0.05:
            fun_message = "Significant! "

        plot3_path = "static/plot3.png"
        plt.figure(figsize=(8, 6))
        plt.hist(simulated_stats, bins=20, alpha=0.7, color='skyblue', edgecolor='black', label="Simulated Statistics")
        if parameter == "slope":
            plt.axvline(observed_stat, color='red', linestyle='--', linewidth=2, label=f"Observed Slope: {observed_stat:.4f}")
        else:
            plt.axvline(observed_stat, color='red', linestyle='--', linewidth=2, label=f"Observed Intercept: {observed_stat:.4f}")
        if parameter == "slope":
            plt.axvline(hypothesized_value, color='purple', linestyle='-', linewidth=2, label=f"Hypothesized Slope (H₀): {hypothesized_value}")
        else:
            plt.axvline(hypothesized_value, color='purple', linestyle='-', linewidth=2, label=f"Hypothesized Intercept (H₀): {hypothesized_value}")

        if parameter == "slope":
            plt.title(f'Hypothesis Test for Slope. p-value: {p_value:.4f}')
            plt.xlabel("Slope")
        else:
            plt.title(f'Hypothesis Test for Intercept. p-value: {p_value:.4f}')
            plt.xlabel("Intercept")
        plt.ylabel("Frequency")

        plt.legend()
        plt.tight_layout()
        plt.savefig(plot3_path)
        plt.close()

        # 返回结果到模板
        return render_template(
            "index.html",
            plot1="static/plot1.png",
            plot2="static/plot2.png",
            plot3=plot3_path,
            parameter=parameter,
            observed_stat=observed_stat,
            hypothesized_value=hypothesized_value,
            N=N,
            beta0=beta0,
            beta1=beta1,
            S=S,
            p_value=p_value,
            fun_message=fun_message,
        )
    except Exception as e:
        print(f"Error in hypothesis_test: {e}")
        return "An error occurred during hypothesis testing.", 500

@app.route("/confidence_interval", methods=["POST"])
def confidence_interval():
    try:
        N = data_store.N
        mu = data_store.mu
        sigma2 = data_store.sigma2
        beta0 = data_store.beta0
        beta1 = data_store.beta1
        S = data_store.S
        X = data_store.X
        Y = data_store.Y
        slope = data_store.slope
        intercept = data_store.intercept
        slopes = data_store.slopes
        intercepts = data_store.intercepts

        parameter = request.form.get("parameter")
        confidence_level = float(request.form.get("confidence_level"))


        if parameter == "slope":
            estimates = np.array(slopes)
            observed_stat = slope
            true_param = beta1
            param_name = "Slope"
        else:
            estimates = np.array(intercepts)
            observed_stat = intercept
            true_param = beta0
            param_name = "Intercept"

        mean_estimate = np.mean(estimates)
        std_estimate = np.std(estimates, ddof=1)

        alpha = 1 - (confidence_level / 100.0)
        z_critical = stats.norm.ppf(1 - alpha / 2)
        margin_of_error = z_critical * (std_estimate / np.sqrt(S))
        ci_lower = mean_estimate - margin_of_error
        ci_upper = mean_estimate + margin_of_error

        includes_true = ci_lower <= true_param <= ci_upper

        plot4_path = "static/plot4.png"
        plt.figure(figsize=(10, 5))  #


        y_coords = np.ones_like(estimates)


        plt.scatter(estimates, y_coords, color='gray', alpha=0.5, label="Estimates")


        plt.scatter(mean_estimate, 1, color='blue' if includes_true else 'red', label="Mean Estimate", zorder=5)


        plt.hlines(1, ci_lower, ci_upper, colors='green', linestyles='solid', linewidth=2, label=f"{int(confidence_level)}% Confidence Interval")


        plt.axvline(true_param, color='purple', linestyle='--', linewidth=2, label="True Parameter")


        plt.title(f"{param_name} Confidence Interval (Confidence Level: {confidence_level}%)")
        plt.xlabel(f"{param_name} Estimates")
        plt.yticks([])
        plt.ylim(0.5, 1.5)
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig(plot4_path)
        plt.close()

        # 返回结果到模板
        return render_template(
            "index.html",
            plot1="static/plot1.png",
            plot2="static/plot2.png",
            plot4=plot4_path,
            parameter=parameter,
            confidence_level=confidence_level,
            mean_estimate=mean_estimate,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            includes_true=includes_true,
            observed_stat=observed_stat,
            N=N,
            mu=mu,
            sigma2=sigma2,
            beta0=beta0,
            beta1=beta1,
            S=S,
        )
    except Exception as e:
        print(f"Error in confidence_interval: {e}")
        return "An error occurred during confidence interval calculation.", 500

if __name__ == "__main__":
    app.run(debug=True)
