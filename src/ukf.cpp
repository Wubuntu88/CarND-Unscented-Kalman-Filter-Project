#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 * This is scaffolding, do not modify
 */
UKF::UKF() {
  cout << "initializing UKF..." << std::endl;
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 1.6;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.7;
  
  //DO NOT MODIFY measurement noise values below these are provided by the sensor manufacturer.
  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;
  //DO NOT MODIFY measurement noise values above these are provided by the sensor manufacturer.

  n_z_lidar_ = 2;
  n_z_radar_ = 3;

  R_laser_ = MatrixXd(n_z_lidar_, n_z_lidar_);
  R_laser_ << std_laspx_ * std_laspx_, 0,
              0, std_laspy_ * std_laspy_;

  R_radar_ = MatrixXd(n_z_radar_, n_z_radar_);
  R_radar_ << std_radr_ * std_radr_, 0, 0,
      0, std_radphi_ * std_radphi_, 0,
      0, 0, std_radrd_ * std_radrd_;

  H_laser_ = MatrixXd(2, 5);
  H_laser_ << 1, 0, 0, 0, 0,
              0, 1, 0, 0, 0;

  is_initialized_ = false;
  time_us_ = 0;
  not_zero = 0.001;
  n_x_ = 5;
  n_aug_ = 7;
  n_p_ = 2 * n_aug_ + 1;
  lambda_ = 3 - n_aug_;

  Xsig_pred_ = MatrixXd(n_aug_, n_p_);

  weights_ = VectorXd(n_p_);
  weights_(0) = lambda_ / (lambda_ + n_aug_);
  for (int i = 1; i < n_p_; ++i) {
    weights_(i) = 0.5 / (n_aug_ + lambda_);
  }

}

UKF::~UKF() {
}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage measurement_pack) {
  /**
   Complete this function! Make sure you switch between lidar and radar
   measurements.
   */
  cout << "Process measurement start" << std::endl;
  if (!is_initialized_) {
    cout << "UKF: " << std::endl;
    x_.fill(0.0);
    P_ = MatrixXd::Identity(n_x_, n_x_);
    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      cout << "Process measurement initializing" << std::endl;
      float rho = measurement_pack.raw_measurements_[0];
      float phi = measurement_pack.raw_measurements_[1];
      float rho_dot = measurement_pack.raw_measurements_[2];

      float px = rho * cos(phi);
      float py = rho * sin(phi);
      float vx = rho_dot * cos(phi);
      float vy = rho_dot * sin(phi);
      float v = sqrt(vx * vx + vy * vy);

      x_ << px, py, v, 0, 0;
    } else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      /**
       Initialize state.  Makes sure the value are not too low
       */
      const double x0 = fabs(measurement_pack.raw_measurements_[0]);
      const double x1 = fabs(measurement_pack.raw_measurements_[1]);
      const double threshold = 0.0001;
      x_(0) = max(x0, threshold);  // x
      x_(1) = max(x1, threshold);  // y
    }

    time_us_ = measurement_pack.timestamp_;
    is_initialized_ = true;
  }
  /*
   * Executed when the UKF is not initialized
   */
  else {
    float dt = (measurement_pack.timestamp_ - time_us_) / 1000000.0;  //convert to seconds
    time_us_ = measurement_pack.timestamp_;

    Prediction(dt);

    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR
        && use_radar_) {
      UpdateRadar(measurement_pack);
    } else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER
        && use_laser_) {
      UpdateLidar(measurement_pack);
    }
  }
  cout << "Process measurement end" << std::endl;
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  /**
   Estimates the object's location. Modifies the state
   vector, x_. Predicts sigma points, the state, and the state covariance matrix.
   */
  MatrixXd Xsig_aug = createSigmaPoints();
  Xsig_pred_ = predictSigmaPoints(Xsig_aug, delta_t);
  predictMeanAndCovariance();
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage measurement_package) {
  cout << "update lidar started" << std::endl;
  /**
   Uses lidar data to update the belief about the object's
   position. Modifies the state vector, x_, and covariance, P_.

   You'll also need to calculate the lidar NIS.
   */
  VectorXd z = measurement_package.raw_measurements_;

  VectorXd y = z - H_laser_ * x_;
  MatrixXd Ht = H_laser_.transpose();
  MatrixXd S = H_laser_ * P_ * Ht + R_laser_;
  MatrixXd Si = S.inverse();
  MatrixXd K = P_ * Ht * Si;

  x_ += K * y;
  x_(3) = normalizeAngle(x_(3)); // essential to normalize; don't remove

  MatrixXd I = MatrixXd::Identity(x_.size(), x_.size());
  P_ = (I - K * H_laser_) * P_;
  cout << "update lidar ended" << std::endl;
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage measurement_package) {
  cout << "update radar started" << std::endl;
  /**
   Uses radar data to update the belief about the object's
   position. Modifies the state vector, x_, and covariance, P_.

   You'll also need to calculate the radar NIS.
   */
  MatrixXd Zsig = MatrixXd(n_z_radar_, n_p_);

  for (int i = 0; i < n_p_; ++i) {
    double p_x = Xsig_pred_(0, i);
    double p_y = Xsig_pred_(1, i);
    double v = Xsig_pred_(2, i);
    double yaw = Xsig_pred_(3, i);

    double v1 = cos(yaw) * v;
    double v2 = sin(yaw) * v;

    Zsig(0, i) = sqrt(p_x * p_x + p_y * p_y);
    Zsig(1, i) = atan2(p_y, p_x);
    if (fabs(Zsig(0, i)) > not_zero) {
      Zsig(2, i) = (p_x * v1 + p_y * v2) / Zsig(0, i);
    } else {
      Zsig(2, i) = 0.0;
    }
  }
  // predict the mean
  VectorXd z_pred = VectorXd(n_z_radar_);
  z_pred.fill(0.0);
  for (int j = 0; j < n_p_; ++j) {
    z_pred += weights_(j) * Zsig.col(j);
  }

  // measurement covariance Matrix S
  MatrixXd S = MatrixXd(n_z_radar_, n_z_radar_);
  S.fill(0.0);
  for (int j = 0; j < n_p_; ++j) {
    VectorXd z_diff = Zsig.col(j) - z_pred;
    z_diff(1) = normalizeAngle(z_diff(1));
    S += weights_(j) * z_diff * z_diff.transpose();
  }
  S += R_radar_;

  // create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z_radar_);
  Tc.fill(0.0);
  for (int j = 0; j < n_p_; ++j) {
    //residual
    VectorXd z_diff = Zsig.col(j) - z_pred;
    z_diff(1) = normalizeAngle(z_diff(1));

    VectorXd x_diff = Xsig_pred_.col(j) - x_;

    x_diff(3) = normalizeAngle(x_diff(3));

    Tc += weights_(j) * x_diff * z_diff.transpose();
  }
  //calculate Kalman gain K
  MatrixXd K = Tc * S.inverse();

  //residual
  VectorXd z_diff = measurement_package.raw_measurements_ - z_pred;

  z_diff(1) = normalizeAngle(z_diff(1));

  x_ += K * z_diff;
  P_ -= K * S * K.transpose();

  cout << "update radar ended" << std::endl;
}

MatrixXd UKF::createSigmaPoints() {
  cout << "creating sigma points started" << std::endl;
  VectorXd x_aug = VectorXd(n_aug_);
  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);
  MatrixXd Xsig_aug = MatrixXd(n_aug_, n_p_);

  // create augmented mean state
  x_aug.fill(0.0);
  x_aug.head(5) = x_;

  // create augmented covariance matrix
  P_aug.fill(0.0);
  P_aug.topLeftCorner(n_x_, n_x_) = P_;
  P_aug(n_x_, n_x_) = std_a_ * std_a_;
  P_aug(n_x_ + 1, n_x_ + 1) = std_yawdd_ * std_yawdd_;

  // create square root matrix
  MatrixXd L = P_aug.llt().matrixL();
  MatrixXd sqrtMtrx = L * sqrt(lambda_ + n_aug_);

  // create augmented sigma points
  Xsig_aug.col(0) = x_aug;
  for (int i = 0; i < n_aug_; ++i) {
    VectorXd sqrtVecColI = sqrtMtrx.col(i);
    Xsig_aug.col(i + 1) = x_aug + sqrtVecColI;
    Xsig_aug.col(i + 1 + n_aug_) = x_aug - sqrtVecColI;
  }
  cout << "creating sigma points ended" << std::endl;
  return Xsig_aug;
}

MatrixXd UKF::predictSigmaPoints(MatrixXd Xsig_aug, const double dt) {
  MatrixXd Xsig_pred_ = MatrixXd(n_x_, n_p_);

  for (int i = 0; i < n_p_; ++i) {
    double p_x = Xsig_aug(0, i);
    double p_y = Xsig_aug(1, i);
    double v = Xsig_aug(2, i);
    double yaw = Xsig_aug(3, i);
    double yawd = Xsig_aug(4, i);
    double nu_a = Xsig_aug(5, i);
    double nu_yawdd = Xsig_aug(6, i);

    double px_p, py_p;

    // avoids division by zero
    if (fabs(yawd) < 0.001) {
      px_p = p_x + v * cos(yaw) * dt;
      py_p = p_y + v * sin(yaw) * dt;
    } else {
      double v_div_yawd = v / yawd;
      double yaw_plus_yawd_times_dt = yaw + yawd * dt;
      px_p = p_x + v_div_yawd * (sin(yaw_plus_yawd_times_dt) - sin(yaw));
      py_p = p_y + v_div_yawd * (cos(yaw) - cos(yaw_plus_yawd_times_dt));
    }

    double dt_sq = dt * dt;

    double v_p = v;
    double yaw_p = yaw + yawd * dt;
    double yawd_p = yawd;

    // adding noise
    px_p += 0.5 * nu_a * dt_sq * cos(yaw);
    py_p += 0.5 * nu_a * dt_sq * sin(yaw);
    v_p += nu_a * dt;

    yaw_p += 0.5 * nu_yawdd * dt_sq;
    yawd_p += nu_yawdd * dt;

    // write predicted sigma point into the column
    Xsig_pred_(0, i) = px_p;
    Xsig_pred_(1, i) = py_p;
    Xsig_pred_(2, i) = v_p;
    Xsig_pred_(3, i) = yaw_p;
    Xsig_pred_(4, i) = yawd_p;
  }

  return Xsig_pred_;
}

void UKF::predictMeanAndCovariance() {
  VectorXd x = VectorXd(n_x_);
  MatrixXd P = MatrixXd(n_x_, n_x_);

  // predict state mean
  x.fill(0.0);
  for (int i = 0; i < n_x_; ++i) {
    x += weights_(i) * Xsig_pred_.col(i);
  }

  // predict covariance matrix
  P.fill(0.0);
  for (int i = 0; i < n_x_; ++i) {
    MatrixXd state_diff = Xsig_pred_.col(i) - x;
    state_diff(3) = normalizeAngle(state_diff(3));
    MatrixXd state_diff_transpose = state_diff.transpose();

    P += weights_(i) * state_diff * state_diff_transpose;
  }
  x_ = x;
  P_ = P;
}

/*
 * makes angle between -pi and pi
 */
double UKF::normalizeAngle(double angle) {
  angle = fmod(angle + M_PI, 2 * M_PI);
  if (angle < 0)
    angle += 2 * M_PI;
  return angle - M_PI;
}
